"""
Voice chat endpoint — Pipecat pipeline over WebSocket.

Uses Deepgram STT, Cartesia TTS, and the same volume-tools agent loop
as the text chat endpoint.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure volume_tools is importable
_API_DIR = str(Path(__file__).resolve().parent)
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputTransportMessageFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

import volume_tools

# -- Config --------------------------------------------------------------------

OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_CHAT_MODEL = os.environ.get("OPENCODE_MODEL_DEFAULT", "deepseek-v4-flash")
MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = (
    "You are a helpful voice assistant that can browse experiment volumes AND "
    "orchestrate live sandboxes running OpenCode agents.\n\n"
    "Capabilities:\n"
    "- Browse stored experiments: list_volumes, list_files, read_file, grep\n"
    "- Manage sandboxes: list_sandboxes, send_to_sandbox, send_to_sandbox_async\n"
    "- Read live sandbox files: list_sandbox_files, read_sandbox_file\n\n"
    "Guidelines:\n"
    "- Keep responses concise and conversational — this is voice, not text.\n"
    "- Summarize rather than reading verbatim.\n"
    "- For active work (run code, edit files), delegate to a sandbox with send_to_sandbox.\n"
    "- Use list_volumes or list_sandboxes first to discover what's available."
)


# -- Custom LLM processor that runs the volume-tools agent loop ----------------


class VolumeToolsLLM(FrameProcessor):
    """Pipecat processor that takes transcription text and runs the volume-tools
    agent loop, then pushes the final text response downstream to TTS."""

    def __init__(self):
        super().__init__()
        self._history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        frame_type = type(frame).__name__
        if frame_type not in ("InputAudioRawFrame", "OutputAudioRawFrame"):
            logger.debug(f"[VolumeToolsLLM] received frame: {frame_type}")

        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text
            logger.info(f"[VolumeToolsLLM] transcription received: '{user_text}'")
            if not user_text or not user_text.strip():
                return

            self._history.append({"role": "user", "content": user_text})

            await self._send_status({"type": "transcription", "text": user_text})

            # Run the tool-calling agent loop
            logger.info("[VolumeToolsLLM] starting agent loop...")
            response_text = await self._run_agent_loop()
            logger.info(f"[VolumeToolsLLM] agent loop complete, response: {response_text[:200]}")

            # Send response text to client as a JSON message (frontend display)
            await self._send_status({"type": "response", "text": response_text})

            # Only the final response text is piped to TTS — thinking and
            # tool-call chatter stays out of audio.
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(text=response_text))
            await self.push_frame(LLMFullResponseEndFrame())

        elif isinstance(frame, EndFrame):
            logger.info("[VolumeToolsLLM] received EndFrame")
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _send_status(self, payload: dict) -> None:
        """Send a JSON status message to the connected frontend (not to TTS)."""
        await self.push_frame(OutputTransportMessageFrame(
            message=json.dumps(payload)
        ))

    async def _run_agent_loop(self) -> str:
        """Run the tool-calling agent loop. Emits thinking / tool_call /
        tool_result events to the frontend as it goes; returns the final
        response text (the only thing piped to TTS)."""
        from openai import OpenAI

        api_key = os.environ.get("OPENCODE_GO_API_KEY", "")
        client = OpenAI(base_url=OPENCODE_ZEN_BASE_URL, api_key=api_key)
        tools_schema = volume_tools.GLOBAL_TOOLS_SCHEMA

        await self._send_status({"type": "thinking"})

        for step in range(MAX_AGENT_STEPS):
            logger.debug(f"[agent] step {step+1}/{MAX_AGENT_STEPS}, history length: {len(self._history)}")
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=DEFAULT_CHAT_MODEL,
                    messages=self._history,
                    tools=tools_schema,
                )
            except Exception as e:
                logger.error(f"[agent] LLM call failed: {type(e).__name__}: {e}")
                return f"Sorry, I hit an error: {e}"

            msg = resp.choices[0].message
            logger.debug(f"[agent] LLM response — content: {bool(msg.content)}, tool_calls: {len(msg.tool_calls) if msg.tool_calls else 0}")

            # Build history entry
            assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
            if getattr(msg, "reasoning_content", None):
                assistant_msg["reasoning_content"] = msg.reasoning_content
                logger.debug(f"[agent] reasoning_content: {msg.reasoning_content[:100]}...")
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self._history.append(assistant_msg)

            if not msg.tool_calls:
                logger.info(f"[agent] final response (no tools): {(msg.content or '')[:100]}")
                return msg.content or "(no response)"

            # Execute tools
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                logger.info(f"[agent] calling tool: {tool_name}({tc.function.arguments})")
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                await self._send_status({
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": args,
                })
                try:
                    result = await asyncio.to_thread(
                        volume_tools.dispatch_global_tool, tool_name, args
                    )
                    result_str = json.dumps(result, default=str)
                    logger.debug(f"[agent] tool {tool_name} result: {result_str[:200]}")
                except Exception as e:
                    logger.error(f"[agent] tool {tool_name} failed: {e}")
                    result_str = json.dumps({"error": f"{type(e).__name__}: {e}"})
                await self._send_status({
                    "type": "tool_result",
                    "name": tool_name,
                    "summary": result_str[:200],
                })
                self._history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        # Hit step budget
        logger.warning(f"[agent] hit step budget ({MAX_AGENT_STEPS})")
        for h in reversed(self._history):
            if h.get("role") == "assistant" and h.get("content"):
                return h["content"]
        return "I ran out of steps trying to answer that."



# -- Pipecat WebSocket server --------------------------------------------------

async def run_voice_pipeline(websocket):
    """Run a Pipecat voice pipeline for a single WebSocket connection.

    Called from the FastAPI WebSocket endpoint in app.py.
    Pipeline: transport → VAD → STT → LLM → TTS → transport
    """
    import aiohttp

    logger.info("[voice] new connection, building pipeline")
    logger.info(f"[voice] STT api_key set: {bool(os.environ.get('ELEVENLABS_API_KEY'))}")
    logger.info(f"[voice] LLM model: {DEFAULT_CHAT_MODEL}, base_url: {OPENCODE_ZEN_BASE_URL}")

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    aiohttp_session = aiohttp.ClientSession()

    stt = ElevenLabsSTTService(
        api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        aiohttp_session=aiohttp_session,
    )

    tts = ElevenLabsTTSService(
        api_key=os.environ.get("ELEVENLABS_API_KEY", ""),
        voice_id=os.environ.get(
            "ELEVENLABS_VOICE_ID",
            "xoq3IlHlZwJusI7LhOD5",  # "Brian"
        ),
    )

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000))
    llm = VolumeToolsLLM()

    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        llm,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    try:
        runner = PipelineRunner()
        await runner.run(task)
    finally:
        await aiohttp_session.close()
        logger.info("[voice] pipeline ended")
