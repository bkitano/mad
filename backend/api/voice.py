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
from pipecat.transports.websocket.server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

import volume_tools

# -- Config --------------------------------------------------------------------

OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_CHAT_MODEL = os.environ.get("OPENCODE_MODEL_DEFAULT", "deepseek-v4-flash")
MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = (
    "You are a helpful voice assistant with tool-access to Modal volumes. "
    "Use `list_volumes` to discover available volumes, then use the other tools "
    "to explore files and answer questions about experiments.\n\n"
    "Guidelines:\n"
    "- Keep responses concise and conversational — this is voice, not text.\n"
    "- Summarize file contents rather than reading them verbatim.\n"
    "- If you need to call tools, do so, then give a spoken summary of what you found.\n"
    "- Use `list_volumes` first if you don't know which volume to look at."
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

            await self.push_frame(OutputTransportMessageFrame(
                message=json.dumps({"type": "transcription", "text": user_text})
            ))

            # Run the tool-calling agent loop
            logger.info("[VolumeToolsLLM] starting agent loop...")
            response_text = await asyncio.to_thread(self._run_agent_loop)
            logger.info(f"[VolumeToolsLLM] agent loop complete, response: {response_text[:200]}")

            # Send response text to client as a JSON message
            await self.push_frame(OutputTransportMessageFrame(
                message=json.dumps({"type": "response", "text": response_text})
            ))

            # Push response to TTS
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(text=response_text))
            await self.push_frame(LLMFullResponseEndFrame())

        elif isinstance(frame, EndFrame):
            logger.info("[VolumeToolsLLM] received EndFrame")
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    def _run_agent_loop(self) -> str:
        """Synchronous agent loop (runs in thread)."""
        from openai import OpenAI

        api_key = os.environ.get("OPENCODE_GO_API_KEY", "")
        client = OpenAI(base_url=OPENCODE_ZEN_BASE_URL, api_key=api_key)
        tools_schema = volume_tools.GLOBAL_TOOLS_SCHEMA

        for step in range(MAX_AGENT_STEPS):
            logger.debug(f"[agent] step {step+1}/{MAX_AGENT_STEPS}, history length: {len(self._history)}")
            try:
                resp = client.chat.completions.create(
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
                    result = volume_tools.dispatch_global_tool(tool_name, args)
                    result_str = json.dumps(result, default=str)
                    logger.debug(f"[agent] tool {tool_name} result: {result_str[:200]}")
                except Exception as e:
                    logger.error(f"[agent] tool {tool_name} failed: {e}")
                    result_str = json.dumps({"error": f"{type(e).__name__}: {e}"})
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

VOICE_WS_PORT = int(os.environ.get("VOICE_WS_PORT", "8765"))


async def start_voice_server():
    """Start the Pipecat WebSocket server on its own port.

    Clients connect to ws://host:VOICE_WS_PORT.
    Pipeline: transport → STT → LLM → TTS → transport
    """
    import aiohttp

    transport = WebsocketServerTransport(
        host="0.0.0.0",
        port=VOICE_WS_PORT,
        params=WebsocketServerParams(
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport_ref, client):
        logger.info(f"[voice] client connected: {client}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport_ref, client):
        logger.info(f"[voice] client disconnected: {client}")

    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport_ref, client):
        logger.warning(f"[voice] session timeout: {client}")

    logger.info(f"[voice] starting WebSocket server on ws://0.0.0.0:{VOICE_WS_PORT}")
    logger.info(f"[voice] pipeline: transport.input → ElevenLabsSTT → VolumeToolsLLM → ElevenLabsTTS → transport.output")
    logger.info(f"[voice] STT api_key set: {bool(os.environ.get('ELEVENLABS_API_KEY'))}")
    logger.info(f"[voice] LLM model: {DEFAULT_CHAT_MODEL}, base_url: {OPENCODE_ZEN_BASE_URL}")

    runner = PipelineRunner()
    await runner.run(task)
