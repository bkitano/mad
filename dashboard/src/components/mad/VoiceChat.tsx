/**
 * Self-contained voice chat panel using the Pipecat React SDK.
 * Completely separate from the text chat (useChat / AI SDK).
 */
import { useCallback, useEffect, useMemo, useRef } from 'react'
import {
  PipecatClientProvider,
  PipecatClientAudio,
  usePipecatClient,
  usePipecatConversation,
  usePipecatClientTransportState,
  usePipecatClientMicControl,
  VoiceVisualizer,
} from '@pipecat-ai/client-react'
import { PipecatClient } from '@pipecat-ai/client-js'
import { WebSocketTransport, ProtobufFrameSerializer } from '@pipecat-ai/websocket-transport'

interface VoiceChatProps {
  wsUrl: string
  onClose: () => void
}

function VoiceChatInner({ wsUrl, onClose }: VoiceChatProps) {
  const client = usePipecatClient()
  const transportState = usePipecatClientTransportState()
  const { isMicEnabled, enableMic } = usePipecatClientMicControl()
  const scrollRef = useRef<HTMLDivElement>(null)

  const { messages } = usePipecatConversation()

  const isConnected = transportState === 'ready'
  const isConnecting = transportState === 'connecting' || transportState === 'connected'

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  const connect = useCallback(async () => {
    if (!client) return
    try {
      await client.connect({ wsUrl })
    } catch (err) {
      console.error('Voice connect failed:', err)
    }
  }, [client, wsUrl])

  const disconnect = useCallback(async () => {
    if (!client) return
    try {
      await client.disconnect()
    } catch { /* best effort */ }
  }, [client])

  return (
    <div className="fixed bottom-6 right-6 z-50 w-96 max-w-[calc(100vw-3rem)] h-[32rem] max-h-[calc(100dvh-3rem)] flex flex-col bg-gray-900 border border-gray-700 rounded-xl shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-200">Voice</span>
          <span className={`text-xs px-1.5 py-0.5 rounded ${
            isConnected ? 'bg-green-900 text-green-300' :
            isConnecting ? 'bg-yellow-900 text-yellow-300' :
            'bg-gray-800 text-gray-400'
          }`}>
            {transportState}
          </span>
        </div>
        <button onClick={onClose} className="p-1 text-gray-400 hover:text-gray-200 cursor-pointer" title="Close">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
        </button>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 && isConnected && (
          <div className="text-sm text-gray-500 text-center mt-8">
            Speak to start a conversation.
          </div>
        )}
        {messages.length === 0 && !isConnected && (
          <div className="text-sm text-gray-500 text-center mt-8">
            Connect to start speaking.
          </div>
        )}
        {messages.filter(m => m.role === 'user' || m.role === 'assistant').map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] px-3 py-2 rounded-lg text-sm whitespace-pre-wrap ${
              m.role === 'user' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-100 border border-gray-700'
            }`}>
              {m.parts.map((part, j) => {
                if (typeof part.text === 'string') return <span key={j}>{part.text}</span>
                // BotOutputText has spoken/unspoken
                const bot = part.text as { spoken?: string; unspoken?: string }
                return <span key={j}>{bot.spoken || bot.unspoken || ''}</span>
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="border-t border-gray-800 p-3 shrink-0">
        {isConnected ? (
          <div className="flex items-center gap-3">
            <div className="flex-1 flex items-center justify-center py-2">
              <VoiceVisualizer
                participantType="local"
                barColor="#a78bfa"
                backgroundColor="transparent"
                barCount={5}
                barWidth={4}
                barGap={3}
                barMaxHeight={24}
              />
            </div>
            <button
              onClick={() => enableMic(!isMicEnabled)}
              className={`p-2.5 rounded-full transition-colors cursor-pointer ${
                isMicEnabled ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-400'
              }`}
              title={isMicEnabled ? 'Mute' : 'Unmute'}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path d="M7 4a3 3 0 0 1 6 0v6a3 3 0 1 1-6 0V4Z" />
                <path d="M5.5 9.643a.75.75 0 0 0-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5h-1.5v-1.546A6.001 6.001 0 0 0 16 10v-.357a.75.75 0 0 0-1.5 0V10a4.5 4.5 0 0 1-9 0v-.357Z" />
              </svg>
            </button>
            <button
              onClick={disconnect}
              className="p-2.5 bg-red-600 hover:bg-red-500 rounded-full text-white transition-colors cursor-pointer"
              title="Disconnect"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path fillRule="evenodd" d="M5.5 3.5A1.5 1.5 0 0 1 7 5v10a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5Zm7 0A1.5 1.5 0 0 1 14 5v10a1.5 1.5 0 0 1-3 0V5a1.5 1.5 0 0 1 1.5-1.5Z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        ) : (
          <button
            onClick={connect}
            disabled={isConnecting}
            className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-medium transition-colors cursor-pointer"
          >
            {isConnecting ? 'Connecting...' : 'Start Voice'}
          </button>
        )}
      </div>

      {/* Hidden audio element for bot playback */}
      <PipecatClientAudio />
    </div>
  )
}

export default function VoiceChat({ wsUrl, onClose }: VoiceChatProps) {
  // Force echo cancellation on all getUserMedia calls to prevent feedback loops
  useEffect(() => {
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
    navigator.mediaDevices.getUserMedia = (constraints) => {
      if (constraints?.audio) {
        constraints.audio = typeof constraints.audio === 'object'
          ? { ...constraints.audio, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
          : { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      }
      return original(constraints)
    }
    return () => { navigator.mediaDevices.getUserMedia = original }
  }, [])

  const client = useMemo(() => new PipecatClient({
    transport: new WebSocketTransport({
      serializer: new ProtobufFrameSerializer(),
    }),
    enableMic: true,
    enableCam: false,
  }), [])

  // Disconnect and clean up when the component unmounts (panel dismissed)
  useEffect(() => {
    return () => {
      try { client.disconnect() } catch { /* best effort */ }
    }
  }, [client])

  return (
    <PipecatClientProvider client={client as any}>
      <VoiceChatInner wsUrl={wsUrl} onClose={onClose} />
    </PipecatClientProvider>
  )
}
