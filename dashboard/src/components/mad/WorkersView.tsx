import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface WorkersViewProps {
  apiUrl: string
}

interface Worker {
  opencode_url: string
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export default function WorkersView({ apiUrl }: WorkersViewProps) {
  const [workers, setWorkers] = useState<Record<string, Worker>>({})
  const [loading, setLoading] = useState(true)
  const [selectedWorker, setSelectedWorker] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Fetch workers
  const fetchWorkers = async () => {
    try {
      const res = await fetch(`${apiUrl}/workers`)
      if (res.ok) {
        setWorkers(await res.json())
      }
    } catch (err) {
      console.error('Error fetching workers:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchWorkers()
    const interval = setInterval(fetchWorkers, 10000)
    return () => clearInterval(interval)
  }, [apiUrl])

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Reset chat when switching workers
  const selectWorker = (workerId: string) => {
    setSelectedWorker(workerId)
    setSessionId(null)
    setMessages([])
    setInput('')
  }

  const sendMessage = async () => {
    if (!input.trim() || !selectedWorker || sending) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage, timestamp: new Date() }])
    setSending(true)

    try {
      const res = await fetch(`${apiUrl}/workers/${selectedWorker}/prompt/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${err.detail || res.statusText}`,
          timestamp: new Date(),
        }])
        return
      }

      const data = await res.json()

      // Save session ID for follow-up messages
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id)
      }

      // Extract text from response parts
      const parts = data.response?.parts || []
      const textParts = parts
        .filter((p: { type: string }) => p.type === 'text')
        .map((p: { text: string }) => p.text)
      const toolParts = parts
        .filter((p: { type: string; tool?: string; state?: { status?: string } }) => p.type === 'tool')
        .map((p: { tool: string; state?: { status?: string } }) => `[${p.tool}: ${p.state?.status || 'unknown'}]`)

      const content = [...toolParts, ...textParts].join('\n\n') || '(empty response)'

      setMessages(prev => [...prev, {
        role: 'assistant',
        content,
        timestamp: new Date(),
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err}`,
        timestamp: new Date(),
      }])
    } finally {
      setSending(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const workerIds = Object.keys(workers)

  if (loading) {
    return (
      <div className="bg-gray-50 p-8 rounded-lg text-center text-gray-500">
        Loading workers...
      </div>
    )
  }

  return (
    <div className="flex gap-6 h-[calc(100vh-280px)] min-h-[500px]">
      {/* Worker list */}
      <div className="w-72 shrink-0 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-700 uppercase tracking-wide">
            Workers ({workerIds.length})
          </h3>
          <button
            onClick={fetchWorkers}
            className="text-xs text-blue-600 hover:text-blue-800"
          >
            Refresh
          </button>
        </div>

        {workerIds.length === 0 ? (
          <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500 text-sm">
            No workers registered
          </div>
        ) : (
          <div className="space-y-2">
            {workerIds.map(id => (
              <button
                key={id}
                onClick={() => selectWorker(id)}
                className={`w-full text-left p-3 rounded-lg border transition-colors ${
                  selectedWorker === id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <span className="font-mono text-xs text-gray-900 truncate">{id}</span>
                </div>
                <div className="mt-1 text-xs text-gray-500 truncate">
                  {workers[id].opencode_url}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Chat panel */}
      <div className="flex-1 flex flex-col border border-gray-200 rounded-lg overflow-hidden">
        {!selectedWorker ? (
          <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
            Select a worker to start chatting
          </div>
        ) : (
          <>
            {/* Chat header */}
            <div className="bg-gray-50 px-4 py-3 border-b border-gray-200 flex items-center justify-between">
              <div>
                <span className="font-mono text-sm text-gray-900">{selectedWorker}</span>
                {sessionId && (
                  <span className="ml-2 text-xs text-gray-400">session: {sessionId.slice(0, 16)}...</span>
                )}
              </div>
              {sessionId && (
                <button
                  onClick={() => { setSessionId(null); setMessages([]) }}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  New session
                </button>
              )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && (
                <div className="text-center text-gray-400 text-sm mt-8">
                  Send a message to {selectedWorker}
                </div>
              )}
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}>
                    {msg.role === 'assistant' ? (
                      <div className="prose prose-sm max-w-none">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    )}
                    <div className={`text-xs mt-1 ${
                      msg.role === 'user' ? 'text-blue-200' : 'text-gray-400'
                    }`}>
                      {msg.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
              {sending && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg px-4 py-2 text-sm text-gray-500">
                    Thinking...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-gray-200 p-3">
              <div className="flex gap-2">
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={`Message ${selectedWorker}...`}
                  rows={1}
                  className="flex-1 resize-none rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={sending}
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || sending}
                  className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Send
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
