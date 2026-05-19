import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport } from 'ai'
import { useNavigate, useParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useAuth } from '../contexts/AuthContext'
import { API_URL, apiFetch } from '../lib/api'

interface ChatSession {
  id: string
  title: string
  type: string
  created_at: string
  updated_at: string
}

export default function ChatPage() {
  const { sessionId: urlSessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const { session: authSession, user, signOut, signInWithGoogle } = useAuth()

  const [chatSessionId, setChatSessionId] = useState<string | null>(urlSessionId || null)
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([])
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [chatInput, setChatInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  const chatTransport = useMemo(() => new DefaultChatTransport({
    api: `${API_URL}/volumes/chat`,
    body: chatSessionId ? { session_id: chatSessionId } : {},
    headers: () => {
      const token = authSession?.access_token
      return token ? { Authorization: `Bearer ${token}` } as Record<string, string> : {} as Record<string, string>
    },
  }), [chatSessionId, authSession])

  const {
    messages,
    sendMessage,
    status,
    setMessages,
  } = useChat({ transport: chatTransport })

  const sending = status === 'submitted' || status === 'streaming'

  // Sync URL param to state
  useEffect(() => {
    if (urlSessionId && urlSessionId !== chatSessionId) {
      setChatSessionId(urlSessionId)
    }
  }, [urlSessionId])

  // Load persisted messages whenever the active session changes
  useEffect(() => {
    if (!chatSessionId) {
      setMessages([])
      return
    }
    let cancelled = false
    ;(async () => {
      try {
        const res = await apiFetch(`/chats/${chatSessionId}`)
        if (!res.ok) {
          if (!cancelled) setMessages([])
          return
        }
        const data = await res.json()
        if (cancelled) return
        const loaded = (data.messages || []).map((m: any) => {
          const parts = Array.isArray(m.parts) && m.parts.length > 0
            ? m.parts
            : [{ type: 'text', text: m.content || '' }]
          return { id: String(m.id), role: m.role, parts }
        })
        setMessages(loaded)
      } catch {
        if (!cancelled) setMessages([])
      }
    })()
    return () => { cancelled = true }
  }, [chatSessionId])

  const fetchHistory = useCallback(async () => {
    try {
      const res = await apiFetch('/chats?limit=50')
      if (res.ok) setChatHistory(await res.json())
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { fetchHistory() }, [fetchHistory])

  // Refresh history when first assistant message arrives on a new chat
  useEffect(() => {
    if (messages.length > 0 && !urlSessionId) {
      fetchHistory()
    }
  }, [messages.length])

  const loadSession = (id: string) => {
    navigate(`/agent/chat/${id}`)
    setChatSessionId(id)
  }

  const startNewChat = () => {
    navigate('/agent/chat')
    setChatSessionId(null)
  }

  const deleteChat = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Delete this chat?')) return
    try {
      await apiFetch(`/chats/${id}`, { method: 'DELETE' })
      if (chatSessionId === id) startNewChat()
      fetchHistory()
    } catch { /* ignore */ }
  }

  const handleSend = () => {
    if (!chatInput.trim() || sending) return
    const text = chatInput.trim()
    setChatInput('')
    sendMessage({ text })
  }

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages, sending])

  const avatar = user?.user_metadata?.avatar_url
  const displayName = user?.user_metadata?.full_name || user?.email

  return (
    <div className="h-[100dvh] flex bg-gray-950 text-gray-100 overflow-hidden">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="md:hidden fixed inset-0 bg-black/60 z-30"
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed md:static inset-y-0 left-0 z-40 w-72 flex flex-col bg-gray-900 border-r border-gray-800 transform transition-transform duration-200 md:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        {/* Sidebar header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <a href="/agent" className="text-sm text-gray-400 hover:text-gray-200 transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 inline mr-1.5 -mt-0.5">
              <path fillRule="evenodd" d="M17 10a.75.75 0 0 1-.75.75H5.612l4.158 3.96a.75.75 0 1 1-1.04 1.08l-5.5-5.25a.75.75 0 0 1 0-1.08l5.5-5.25a.75.75 0 1 1 1.04 1.08L5.612 9.25H16.25A.75.75 0 0 1 17 10Z" clipRule="evenodd" />
            </svg>
            Dashboard
          </a>
          <button
            onClick={startNewChat}
            className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg transition-colors cursor-pointer"
            title="New chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M5.433 13.917l1.262-3.155A4 4 0 0 1 7.58 9.42l6.92-6.918a2.121 2.121 0 0 1 3 3l-6.92 6.918c-.383.383-.84.685-1.343.886l-3.154 1.262a.5.5 0 0 1-.65-.65Z" />
              <path d="M3.5 5.75c0-.69.56-1.25 1.25-1.25H10A.75.75 0 0 0 10 3H4.75A2.75 2.75 0 0 0 2 5.75v9.5A2.75 2.75 0 0 0 4.75 18h9.5A2.75 2.75 0 0 0 17 15.25V10a.75.75 0 0 0-1.5 0v5.25c0 .69-.56 1.25-1.25 1.25h-9.5c-.69 0-1.25-.56-1.25-1.25v-9.5Z" />
            </svg>
          </button>
        </div>

        {/* Chat list */}
        <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
          {chatHistory.map((s) => (
            <button
              key={s.id}
              onClick={() => loadSession(s.id)}
              className={`group w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors cursor-pointer ${
                chatSessionId === s.id
                  ? 'bg-gray-800 text-gray-100'
                  : 'text-gray-400 hover:bg-gray-800/60 hover:text-gray-200'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate flex-1">{s.title}</span>
                <button
                  onClick={(e) => deleteChat(s.id, e)}
                  className="hidden group-hover:block p-0.5 text-gray-500 hover:text-red-400 cursor-pointer shrink-0"
                  title="Delete"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
                    <path fillRule="evenodd" d="M5 3.25V4H2.75a.75.75 0 0 0 0 1.5h.3l.815 8.15A1.5 1.5 0 0 0 5.357 15h5.285a1.5 1.5 0 0 0 1.493-1.35l.815-8.15h.3a.75.75 0 0 0 0-1.5H11v-.75A2.25 2.25 0 0 0 8.75 1h-1.5A2.25 2.25 0 0 0 5 3.25Zm2.25-.75a.75.75 0 0 0-.75.75V4h3v-.75a.75.75 0 0 0-.75-.75h-1.5ZM6.05 6a.75.75 0 0 1 .787.713l.275 5.5a.75.75 0 0 1-1.498.075l-.275-5.5A.75.75 0 0 1 6.05 6Zm3.9 0a.75.75 0 0 1 .712.787l-.275 5.5a.75.75 0 0 1-1.498-.075l.275-5.5a.75.75 0 0 1 .786-.712Z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
              <div className="text-xs text-gray-500 mt-0.5">
                {s.type === 'voice' ? 'Voice' : 'Text'} &middot; {new Date(s.updated_at).toLocaleDateString()}
              </div>
            </button>
          ))}
          {chatHistory.length === 0 && (
            <p className="text-sm text-gray-500 px-3 py-4 text-center">No chats yet</p>
          )}
        </div>

        {/* Account */}
        <div className="border-t border-gray-800">
          {user ? (
            <div className="flex items-center gap-3 px-3 py-3">
              {avatar ? (
                <img src={avatar} alt="" className="w-7 h-7 rounded-full shrink-0" />
              ) : (
                <div className="w-7 h-7 rounded-full bg-gray-700 flex items-center justify-center text-xs font-medium text-gray-300 shrink-0">
                  {(displayName || '?')[0].toUpperCase()}
                </div>
              )}
              <span className="text-sm text-gray-300 truncate flex-1">{displayName}</span>
              <button onClick={signOut} className="text-xs text-gray-500 hover:text-gray-300 cursor-pointer">Sign out</button>
            </div>
          ) : (
            <div className="p-3">
              <button
                onClick={signInWithGoogle}
                className="w-full py-2 px-3 rounded-lg text-sm text-gray-300 hover:bg-gray-800 border border-gray-700 transition-colors cursor-pointer"
              >
                Sign in
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        {/* Top bar (mobile toggle) */}
        <header className="md:hidden flex items-center gap-3 px-3 py-2 bg-gray-900 border-b border-gray-800 shrink-0">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 text-gray-400 hover:text-gray-200 cursor-pointer"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path fillRule="evenodd" d="M2 4.75A.75.75 0 0 1 2.75 4h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 4.75ZM2 10a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 10Zm0 5.25a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1-.75-.75Z" clipRule="evenodd" />
            </svg>
          </button>
          <span className="text-sm text-gray-400">Chat</span>
        </header>

        {/* Messages area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {messages.length === 0 && !sending && (
              <div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
                <div className="w-12 h-12 rounded-full bg-purple-600/20 flex items-center justify-center mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-purple-400">
                    <path fillRule="evenodd" d="M4.804 21.644A6.707 6.707 0 0 0 6 21.75a6.721 6.721 0 0 0 3.583-1.029c.774.182 1.584.279 2.417.279 5.322 0 9.75-3.97 9.75-9 0-5.03-4.428-9-9.75-9s-9.75 3.97-9.75 9c0 2.409 1.025 4.587 2.674 6.192.232.226.277.428.254.543a3.73 3.73 0 0 1-.814 1.686.75.75 0 0 0 .44 1.223Z" clipRule="evenodd" />
                  </svg>
                </div>
                <h2 className="text-lg font-medium text-gray-200 mb-2">Start a conversation</h2>
                <p className="text-sm text-gray-500 max-w-md">
                  Ask about any volume — the agent can discover and browse files, read notebooks, grep for patterns, and more.
                </p>
              </div>
            )}

            {messages.map((m) => (
              <div key={m.id} className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}>
                {m.role === 'assistant' && (
                  <div className="w-7 h-7 rounded-full bg-purple-600/20 flex items-center justify-center shrink-0 mt-0.5">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-purple-400">
                      <path fillRule="evenodd" d="M10 1a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 10 1ZM5.05 3.05a.75.75 0 0 1 1.06 0l1.062 1.06a.75.75 0 1 1-1.06 1.061L5.05 4.11a.75.75 0 0 1 0-1.06ZM14.95 3.05a.75.75 0 0 1 0 1.06l-1.06 1.062a.75.75 0 0 1-1.062-1.061l1.061-1.06a.75.75 0 0 1 1.06 0ZM3 8a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5h-1.5A.75.75 0 0 1 3 8ZM14 8a.75.75 0 0 1 .75-.75h1.5a.75.75 0 0 1 0 1.5h-1.5A.75.75 0 0 1 14 8ZM7.172 11.828a.75.75 0 0 1 0 1.061l-1.06 1.06a.75.75 0 0 1-1.06-1.06l1.06-1.06a.75.75 0 0 1 1.06 0ZM12.828 11.828a.75.75 0 0 1 1.061 0l1.06 1.06a.75.75 0 0 1-1.06 1.061l-1.06-1.06a.75.75 0 0 1 0-1.061ZM10 14a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 10 14Z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}
                <div className={`${m.role === 'user' ? 'max-w-[70%] bg-purple-600 rounded-2xl rounded-br-md px-4 py-2.5' : 'flex-1 min-w-0'}`}>
                  {m.parts.map((part, j) => {
                    if (part.type === 'text') {
                      return (
                        <div key={j} className={`prose prose-sm max-w-none ${
                          m.role === 'user'
                            ? 'prose-invert text-white prose-p:my-0'
                            : 'prose-invert prose-p:my-2 prose-pre:my-3 prose-ul:my-2 prose-ol:my-2 prose-li:my-0 prose-headings:my-3 prose-code:text-purple-300 prose-a:text-purple-400'
                        }`}>
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              code({ className, children, ...props }) {
                                const match = /language-(\w+)/.exec(className || '')
                                const inline = !match
                                return inline ? (
                                  <code className={className} {...props}>{children}</code>
                                ) : (
                                  <SyntaxHighlighter
                                    style={vscDarkPlus}
                                    language={match[1]}
                                    PreTag="div"
                                    customStyle={{ margin: 0, borderRadius: '0.5rem', fontSize: '13px' }}
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                )
                              }
                            }}
                          >
                            {part.text}
                          </ReactMarkdown>
                        </div>
                      )
                    }
                    if (part.type === 'reasoning') {
                      return (
                        <div key={j} className="text-xs text-gray-500 italic border-l-2 border-gray-700 pl-3 my-2 py-1">
                          {part.text}
                        </div>
                      )
                    }
                    if (part.type.startsWith('tool-')) {
                      const toolPart = part as any
                      return (
                        <div key={j} className="flex items-center gap-2 text-xs font-mono text-gray-400 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 my-2">
                          <span className="text-blue-400">{toolPart.toolName || 'tool'}</span>
                          {toolPart.state === 'result' || toolPart.state === 'output-available' ? (
                            <span className="text-green-400">done</span>
                          ) : (
                            <span className="text-yellow-400 animate-pulse">running...</span>
                          )}
                        </div>
                      )
                    }
                    return null
                  })}
                </div>
              </div>
            ))}

            {status === 'submitted' && (
              <div className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-purple-600/20 flex items-center justify-center shrink-0">
                  <div className="w-3.5 h-3.5 border-2 border-gray-600 border-t-purple-400 rounded-full animate-spin" />
                </div>
                <span className="text-sm text-gray-400 mt-1">Thinking...</span>
              </div>
            )}
          </div>
        </div>

        {/* Input area */}
        <div className="border-t border-gray-800 bg-gray-950 shrink-0">
          <div className="max-w-3xl mx-auto px-4 py-4">
            <div className="relative">
              <textarea
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSend()
                  }
                }}
                placeholder="Ask about your volumes, data, experiments..."
                rows={1}
                disabled={sending}
                className="w-full px-4 py-3 pr-12 bg-gray-900 border border-gray-700 rounded-xl text-sm resize-none focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500/30 disabled:opacity-60 placeholder-gray-500"
                style={{ minHeight: '48px', maxHeight: '200px' }}
                onInput={(e) => {
                  const target = e.target as HTMLTextAreaElement
                  target.style.height = 'auto'
                  target.style.height = Math.min(target.scrollHeight, 200) + 'px'
                }}
              />
              <button
                onClick={handleSend}
                disabled={sending || !chatInput.trim()}
                className="absolute right-2 bottom-2 p-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg transition-colors cursor-pointer"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                  <path d="M3.105 2.288a.75.75 0 0 0-.826.95l1.414 4.926A1.5 1.5 0 0 0 5.135 9.25h6.115a.75.75 0 0 1 0 1.5H5.135a1.5 1.5 0 0 0-1.442 1.086l-1.414 4.926a.75.75 0 0 0 .826.95 28.897 28.897 0 0 0 15.293-7.155.75.75 0 0 0 0-1.114A28.897 28.897 0 0 0 3.105 2.288Z" />
                </svg>
              </button>
            </div>
            <p className="text-xs text-gray-600 mt-2 text-center">
              The agent can browse volumes, read files, search with grep, and inspect notebooks.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
