import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport } from 'ai'
import { useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import NotebookViewer from '../components/mad/NotebookViewer'
import VoiceChat from '../components/mad/VoiceChat'
import { useAuth } from '../contexts/AuthContext'
import { API_URL, apiFetch } from '../lib/api'

interface Session {
  sandbox_id: string
  volume_name: string
  opencode_url: string
  jupyter_url: string
  status: string
  gpu: string
  cpu: number
  memory: number
  expires_at: number  // unix epoch seconds
}

interface SandboxListItem {
  sandbox_id: string
  opencode_url: string | null
  jupyter_url: string | null
  volume_name: string
  gpu: string
  cpu: number
  memory: number
  expires_at: number
}

interface VolumeItem {
  name: string
  volume_id: string
  created_at: string | null
  created_by: string | null
}

type SidebarTab = 'sandboxes' | 'volumes' | 'chats'

function Countdown({ expiresAt }: { expiresAt: number }) {
  const [now, setNow] = useState(() => Math.floor(Date.now() / 1000))
  useEffect(() => {
    const id = setInterval(() => setNow(Math.floor(Date.now() / 1000)), 60_000)
    return () => clearInterval(id)
  }, [])
  if (!expiresAt) return null
  const remaining = expiresAt - now
  if (remaining <= 0) return <span className="text-xs text-red-400">expired</span>
  const hours = Math.floor(remaining / 3600)
  const minutes = Math.floor((remaining % 3600) / 60)
  const isLow = remaining < 1800 // < 30 min
  return (
    <span className={`text-xs ${isLow ? 'text-orange-400' : 'text-gray-500'}`}>
      {hours}h {minutes}m left
    </span>
  )
}

function SidebarAccount() {
  const { user, signOut, signInWithGoogle } = useAuth()
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [open])

  if (!user) {
    return (
      <div className="p-3 border-t border-gray-800">
        <button
          onClick={signInWithGoogle}
          className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-sm text-gray-300 hover:bg-gray-800 border border-gray-700 transition-colors cursor-pointer"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24">
            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4" />
            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
          </svg>
          Sign in with Google
        </button>
      </div>
    )
  }

  const avatar = user.user_metadata?.avatar_url
  const name = user.user_metadata?.full_name || user.email
  const email = user.email

  return (
    <div className="relative border-t border-gray-800" ref={menuRef}>
      {open && (
        <div className="absolute bottom-full left-0 right-0 mb-1 mx-2 bg-gray-800 rounded-lg border border-gray-700 shadow-xl overflow-hidden">
          <div className="px-3 py-2.5 border-b border-gray-700">
            <p className="text-sm font-medium text-gray-200 truncate">{name}</p>
            {email && <p className="text-xs text-gray-500 truncate mt-0.5">{email}</p>}
          </div>
          <button
            onClick={() => { signOut(); setOpen(false) }}
            className="w-full text-left px-3 py-2.5 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-700 transition-colors cursor-pointer"
          >
            Sign out
          </button>
        </div>
      )}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-3 py-3 hover:bg-gray-800 transition-colors cursor-pointer"
      >
        {avatar ? (
          <img src={avatar} alt="" className="w-7 h-7 rounded-full shrink-0" />
        ) : (
          <div className="w-7 h-7 rounded-full bg-gray-700 flex items-center justify-center text-xs font-medium text-gray-300 shrink-0">
            {(name || '?')[0].toUpperCase()}
          </div>
        )}
        <span className="text-sm text-gray-300 truncate flex-1 text-left">{name}</span>
        <svg className="w-3.5 h-3.5 text-gray-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
        </svg>
      </button>
    </div>
  )
}

function MobileMenuButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="md:hidden -ml-1 p-2 text-gray-400 hover:text-gray-200 cursor-pointer shrink-0"
      aria-label="Open sidebar"
    >
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
        <path fillRule="evenodd" d="M2 4.75A.75.75 0 0 1 2.75 4h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 4.75ZM2 10a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 10Zm0 5.25a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1-.75-.75Z" clipRule="evenodd" />
      </svg>
    </button>
  )
}

export default function MADDashboard() {
  const navigate = useNavigate()
  const [sandboxes, setSandboxes] = useState<SandboxListItem[]>([])
  const [session, setSession] = useState<Session | null>(null)
  const [spawning, setSpawning] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [activeTab, setActiveTab] = useState<'opencode' | 'jupyter'>('opencode')
  const [jupyterReady, setJupyterReady] = useState(false)
  const jupyterPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('sandboxes')
  const [volumes, setVolumes] = useState<VolumeItem[]>([])
  const [selectedVolume, setSelectedVolume] = useState<VolumeItem | null>(null)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)

  const [volumePath, setVolumePath] = useState('/')
  const [volumeEntries, setVolumeEntries] = useState<{ path: string; type: string }[]>([])
  const [volumeLoading, setVolumeLoading] = useState(false)
  const [viewingFile, setViewingFile] = useState<{ path: string; content: string; encoding: string } | null>(null)
  const [fileLoading, setFileLoading] = useState(false)

  const [renamingVolume, setRenamingVolume] = useState(false)
  const [renameValue, setRenameValue] = useState('')
  const [renameLoading, setRenameLoading] = useState(false)

  const [chatOpen, setChatOpen] = useState(false)
  const [chatSessionId, setChatSessionId] = useState<string | null>(null)
  const [chatHistory, setChatHistory] = useState<{id: string, title: string, type: string, updated_at: string}[]>([])
  const chatScrollRef = useRef<HTMLDivElement>(null)

  const { session: authSession } = useAuth()

  const chatTransport = useMemo(() => new DefaultChatTransport({
    api: `${API_URL}/volumes/chat`,
    body: chatSessionId ? { session_id: chatSessionId } : {},
    headers: () => {
      const token = authSession?.access_token
      return token ? { Authorization: `Bearer ${token}` } as Record<string, string> : {} as Record<string, string>
    },
  }), [chatSessionId, authSession])

  const {
    messages: chatMessages,
    sendMessage: sendChatMessage,
    status: chatStatus,
    setMessages: setChatMessages,
  } = useChat({ transport: chatTransport })

  const [chatInput, setChatInput] = useState('')
  const chatSending = chatStatus === 'submitted' || chatStatus === 'streaming'

  const fetchChatHistory = async () => {
    try {
      const res = await apiFetch('/chats?limit=20')
      if (res.ok) setChatHistory(await res.json())
    } catch { /* ignore */ }
  }

  const startNewChat = () => {
    setChatSessionId(null)
    setChatMessages([])
  }

  useEffect(() => {
    if (chatMessages.length > 0 && !chatSessionId) {
      fetchChatHistory()
    }
  }, [chatMessages.length])

  // Voice panel state
  const [voiceOpen, setVoiceOpen] = useState(false)

  // Create form state
  const [githubRepo, setGithubRepo] = useState('bkitano/mad-experiments-template')
  const [githubRef, setGithubRef] = useState('main')
  const [volumeName, setVolumeName] = useState('')
  const [gpu, setGpu] = useState('')
  const [gpuCount, setGpuCount] = useState(1)
  const [cpu, setCpu] = useState(4)
  const [memory, setMemory] = useState(8192)

  const pollJupyter = useCallback((url: string) => {
    setJupyterReady(false)
    if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
    jupyterPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(url, { method: 'HEAD', mode: 'no-cors' })
        if (res.type === 'opaque' || res.ok) {
          setJupyterReady(true)
          if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
        }
      } catch {
        // not ready yet
      }
    }, 3000)
  }, [])

  useEffect(() => {
    return () => { if (jupyterPollRef.current) clearInterval(jupyterPollRef.current) }
  }, [])

  useEffect(() => {
    if (session?.jupyter_url) {
      pollJupyter(session.jupyter_url)
    } else {
      setJupyterReady(false)
      if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
    }
  }, [session?.jupyter_url, pollJupyter])

  const fetchSandboxes = async () => {
    try {
      const res = await apiFetch('/sandboxes')
      if (res.ok) {
        const data = await res.json()
        setSandboxes(data.sandboxes || [])
      }
    } catch { /* silently fail */ }
  }

  const fetchVolumes = async () => {
    try {
      const res = await apiFetch('/volumes')
      if (res.ok) {
        const data = await res.json()
        setVolumes(data.volumes || [])
      }
    } catch { /* silently fail */ }
  }

  useEffect(() => {
    fetchSandboxes()
    fetchVolumes()
    const interval = setInterval(fetchSandboxes, 15000)
    return () => clearInterval(interval)
  }, [])

  const spawnSession = async () => {
    setSpawning(true)
    try {
      const body: Record<string, unknown> = {}
      if (githubRepo) body.github_repo = githubRepo
      if (githubRef) body.github_ref = githubRef
      if (volumeName) body.volume_name = volumeName
      if (gpu) body.gpu = gpuCount > 1 ? `${gpu}:${gpuCount}` : gpu
      body.cpu = cpu
      body.memory = memory

      const res = await apiFetch('/sandboxes/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setSession({
        ...data,
        gpu: gpu ? (gpuCount > 1 ? `${gpu}:${gpuCount}` : gpu) : '',
        cpu,
        memory,
        expires_at: Math.floor(Date.now() / 1000) + 6 * 3600,
      } as Session)
      setShowCreateForm(false)
      setSelectedVolume(null)
      setSidebarTab('sandboxes')
      setMobileSidebarOpen(false)
      fetchSandboxes()
    } catch (err) {
      alert(`Failed to spawn session: ${err}`)
    } finally {
      setSpawning(false)
    }
  }

  const selectSandbox = (sb: SandboxListItem) => {
    if (!sb.opencode_url) return
    setSelectedVolume(null)
    setSession({
      sandbox_id: sb.sandbox_id,
      volume_name: sb.volume_name || '',
      opencode_url: sb.opencode_url,
      jupyter_url: sb.jupyter_url || '',
      status: 'running',
      gpu: sb.gpu || '',
      cpu: sb.cpu || 4,
      memory: sb.memory || 8192,
      expires_at: sb.expires_at || 0,
    })
    setMobileSidebarOpen(false)
  }

  const terminateSession = async () => {
    if (!session) return
    if (!confirm('Terminate this session?')) return
    try {
      await apiFetch('/sandboxes/terminate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sandbox_id: session.sandbox_id }),
      })
      setSession(null)
      fetchSandboxes()
    } catch (err) {
      alert(`Failed to terminate: ${err}`)
    }
  }

  const exportFiles = () => {
    if (!session) return
    window.open(session.jupyter_url, '_blank')
  }

  const fetchVolumeContents = async (volName: string, path: string) => {
    setVolumeLoading(true)
    try {
      const params = new URLSearchParams({ volume_name: volName, path })
      const res = await apiFetch(`/volumes/ls?${params}`)
      if (res.ok) {
        const data = await res.json()
        setVolumeEntries(data.entries || [])
      }
    } catch {
      setVolumeEntries([])
    } finally {
      setVolumeLoading(false)
    }
  }

  const handleSelectVolume = (vol: VolumeItem) => {
    setSelectedVolume(vol)
    setSession(null)
    setVolumePath('/')
    setViewingFile(null)
    fetchVolumeContents(vol.name, '/')
    setMobileSidebarOpen(false)
  }

  const handleSendChat = () => {
    if (!chatInput.trim() || chatSending) return
    const text = chatInput.trim()
    setChatInput('')
    sendChatMessage({ text })
  }


  useEffect(() => {
    if (chatScrollRef.current) chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight
  }, [chatMessages, chatSending])

  const voiceToken = authSession?.access_token || ''
  const VOICE_WS_URL = API_URL.replace(/^http/, 'ws') + '/ws/voice' + (voiceToken ? `?token=${voiceToken}` : '')

  const navigateVolume = (path: string) => {
    if (!selectedVolume) return
    setVolumePath(path)
    setViewingFile(null)
    fetchVolumeContents(selectedVolume.name, path)
  }

  const openFile = async (filePath: string) => {
    if (!selectedVolume) return
    setFileLoading(true)
    try {
      const params = new URLSearchParams({ volume_name: selectedVolume.name, path: filePath })
      const res = await apiFetch(`/volumes/read?${params}`)
      if (res.ok) {
        const data = await res.json()
        setViewingFile({ path: data.path, content: data.content, encoding: data.encoding })
      }
    } catch {
      setViewingFile(null)
    } finally {
      setFileLoading(false)
    }
  }

  const handleAttachAndCreate = (vol: VolumeItem) => {
    setVolumeName(vol.name)
    setShowCreateForm(true)
  }

  const startRename = () => {
    if (!selectedVolume) return
    setRenameValue(selectedVolume.name)
    setRenamingVolume(true)
  }

  const submitRename = async () => {
    if (!selectedVolume || !renameValue.trim() || renameValue === selectedVolume.name) {
      setRenamingVolume(false)
      return
    }
    setRenameLoading(true)
    try {
      const res = await apiFetch('/volumes/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_name: selectedVolume.name, new_name: renameValue.trim() }),
      })
      if (!res.ok) throw new Error(await res.text())
      setSelectedVolume({ ...selectedVolume, name: renameValue.trim() })
      setRenamingVolume(false)
      fetchVolumes()
    } catch (err) {
      alert(`Failed to rename: ${err}`)
    } finally {
      setRenameLoading(false)
    }
  }

  const deleteVolume = async () => {
    if (!selectedVolume) return
    if (!confirm(`Delete volume "${selectedVolume.name}"? This is irreversible.`)) return
    try {
      const res = await apiFetch('/volumes/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ volume_name: selectedVolume.name }),
      })
      if (!res.ok) throw new Error(await res.text())
      setSelectedVolume(null)
      fetchVolumes()
    } catch (err) {
      alert(`Failed to delete: ${err}`)
    }
  }

  return (
    <div className="h-[100dvh] flex bg-gray-950 text-gray-100 overflow-hidden">
      {/* Mobile sidebar backdrop */}
      {mobileSidebarOpen && (
        <div
          onClick={() => setMobileSidebarOpen(false)}
          className="md:hidden fixed inset-0 bg-black/60 z-30"
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-64 flex flex-col border-r border-gray-800 bg-gray-900 transform transition-transform duration-200 md:translate-x-0 ${
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex border-b border-gray-800">
          <button
            onClick={() => setSidebarTab('sandboxes')}
            className={`flex-1 py-3 text-sm font-medium transition-colors cursor-pointer ${
              sidebarTab === 'sandboxes'
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Sandboxes
          </button>
          <button
            onClick={() => { setSidebarTab('volumes'); fetchVolumes() }}
            className={`flex-1 py-3 text-sm font-medium transition-colors cursor-pointer ${
              sidebarTab === 'volumes'
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Volumes
          </button>
          <button
            onClick={() => { setSidebarTab('chats'); fetchChatHistory() }}
            className={`flex-1 py-3 text-sm font-medium transition-colors cursor-pointer ${
              sidebarTab === 'chats'
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Chats
          </button>
        </div>

        {sidebarTab === 'sandboxes' && (
          <>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {sandboxes.map((sb) => (
                <button
                  key={sb.sandbox_id}
                  onClick={() => selectSandbox(sb)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
                    session?.sandbox_id === sb.sandbox_id
                      ? 'bg-purple-600/20 text-purple-300 border border-purple-700'
                      : 'hover:bg-gray-800 text-gray-300'
                  }`}
                >
                  <div className="font-mono text-xs truncate">{sb.sandbox_id.slice(0, 16)}</div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    {sb.opencode_url ? 'running' : 'no tunnel'}
                  </div>
                </button>
              ))}
              {sandboxes.length === 0 && (
                <p className="text-sm text-gray-500 px-3 py-2">No running sandboxes</p>
              )}
            </div>
            <div className="p-3 border-t border-gray-800">
              <button
                onClick={() => { fetchVolumes(); setShowCreateForm(true) }}
                className="w-full py-2 bg-purple-600 hover:bg-purple-500 rounded-lg text-sm font-medium transition-colors cursor-pointer"
              >
                Create Sandbox
              </button>
            </div>
          </>
        )}

        {sidebarTab === 'volumes' && (
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {volumes.map((vol) => (
              <button
                key={vol.volume_id}
                onClick={() => handleSelectVolume(vol)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer ${
                  selectedVolume?.volume_id === vol.volume_id
                    ? 'bg-purple-600/20 text-purple-300 border border-purple-700'
                    : 'hover:bg-gray-800 text-gray-300'
                }`}
              >
                <div className="font-mono text-xs truncate">{vol.name}</div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {vol.created_at ? new Date(vol.created_at).toLocaleDateString() : 'unknown date'}
                </div>
              </button>
            ))}
            {volumes.length === 0 && (
              <p className="text-sm text-gray-500 px-3 py-2">No volumes found</p>
            )}
          </div>
        )}

        {sidebarTab === 'chats' && (
          <>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {chatHistory.map((s) => (
                <button
                  key={s.id}
                  onClick={() => navigate(`/agent/chat/${s.id}`)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer hover:bg-gray-800 text-gray-300`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs">{s.type === 'voice' ? '\uD83C\uDF99\uFE0F' : '\uD83D\uDCAC'}</span>
                    <span className="text-xs truncate flex-1">{s.title}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    {new Date(s.updated_at).toLocaleString()}
                  </div>
                </button>
              ))}
              {chatHistory.length === 0 && (
                <p className="text-sm text-gray-500 px-3 py-2">No chats yet</p>
              )}
            </div>
            <div className="p-3 border-t border-gray-800">
              <button
                onClick={() => navigate('/agent/chat')}
                className="w-full py-2 bg-purple-600 hover:bg-purple-500 rounded-lg text-sm font-medium transition-colors cursor-pointer"
              >
                New Chat
              </button>
            </div>
          </>
        )}

        <SidebarAccount />
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        {session ? (
          <>
            <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between px-3 py-2 md:px-4 bg-gray-900 border-b border-gray-800 shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <MobileMenuButton onClick={() => setMobileSidebarOpen(true)} />
                <span className="text-sm font-mono text-gray-400 truncate">{session.sandbox_id.slice(0, 12)}</span>
                <span className="text-xs px-2 py-0.5 bg-green-900 text-green-300 rounded shrink-0">{session.status}</span>
                {session.volume_name && (
                  <span className="hidden md:inline text-xs text-gray-500 truncate">vol: {session.volume_name}</span>
                )}
                <span className="hidden md:inline text-xs text-gray-500 truncate">
                  {session.gpu || 'CPU'} &middot; {session.cpu} cores &middot; {session.memory >= 1024 ? `${(session.memory / 1024).toFixed(0)} GiB` : `${session.memory} MiB`}
                </span>
                <span className="hidden md:inline"><Countdown expiresAt={session.expires_at} /></span>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex bg-gray-800 rounded-lg p-0.5">
                  <button
                    onClick={() => setActiveTab('opencode')}
                    className={`px-3 py-1 text-sm rounded-md transition-colors cursor-pointer ${
                      activeTab === 'opencode' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    OpenCode
                  </button>
                  <button
                    onClick={() => setActiveTab('jupyter')}
                    className={`px-3 py-1 text-sm rounded-md transition-colors cursor-pointer ${
                      activeTab === 'jupyter' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Jupyter
                  </button>
                </div>
                <button onClick={exportFiles} className="px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg transition-colors cursor-pointer">Export</button>
                <button onClick={terminateSession} className="px-3 py-1.5 text-sm bg-red-900 hover:bg-red-800 text-red-200 border border-red-800 rounded-lg transition-colors cursor-pointer">Terminate</button>
              </div>
            </header>
            <main className="flex-1 relative">
              <iframe src={session.opencode_url} className={`absolute inset-0 w-full h-full border-0 ${activeTab === 'opencode' ? 'block' : 'hidden'}`} title="OpenCode" />
              {activeTab === 'jupyter' && !jupyterReady && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950 z-10">
                  <div className="animate-spin rounded-full h-10 w-10 border-2 border-gray-700 border-t-purple-500 mb-4"></div>
                  <p className="text-sm text-gray-400">Waiting for Jupyter to start...</p>
                </div>
              )}
              <iframe src={jupyterReady ? session.jupyter_url : undefined} className={`absolute inset-0 w-full h-full border-0 ${activeTab === 'jupyter' ? 'visible' : 'invisible pointer-events-none'}`} title="Jupyter" />
            </main>
          </>
        ) : selectedVolume ? (
          <div className="flex-1 flex flex-col min-h-0">
            <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between px-3 py-2 md:px-4 bg-gray-900 border-b border-gray-800 shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <MobileMenuButton onClick={() => setMobileSidebarOpen(true)} />
                {renamingVolume ? (
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <input
                      type="text" value={renameValue} onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') submitRename(); if (e.key === 'Escape') setRenamingVolume(false) }}
                      autoFocus disabled={renameLoading}
                      className="flex-1 min-w-0 sm:w-64 sm:flex-none px-2 py-1 bg-gray-800 border border-gray-600 rounded text-sm font-mono text-gray-200 focus:outline-none focus:border-purple-500"
                    />
                    <button onClick={submitRename} disabled={renameLoading} className="px-2 py-1 text-xs bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 rounded transition-colors cursor-pointer shrink-0">{renameLoading ? '...' : 'Save'}</button>
                    <button onClick={() => setRenamingVolume(false)} disabled={renameLoading} className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200 cursor-pointer shrink-0">Cancel</button>
                  </div>
                ) : (
                  <>
                    <span className="text-sm font-mono text-gray-300 truncate">{selectedVolume.name}</span>
                    <button onClick={startRename} className="text-gray-500 hover:text-gray-300 cursor-pointer shrink-0" title="Rename volume">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                        <path d="M2.695 14.763l-1.262 3.154a.5.5 0 00.65.65l3.155-1.262a4 4 0 001.343-.885L17.5 5.5a2.121 2.121 0 00-3-3L3.58 13.42a4 4 0 00-.885 1.343z" />
                      </svg>
                    </button>
                    <span className="hidden md:inline text-xs text-gray-500 shrink-0">
                      {selectedVolume.created_at ? new Date(selectedVolume.created_at).toLocaleString() : ''}
                    </span>
                  </>
                )}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <button onClick={() => handleAttachAndCreate(selectedVolume)} className="px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-500 rounded-lg font-medium transition-colors cursor-pointer">
                  <span className="hidden sm:inline">Create Sandbox with this Volume</span>
                  <span className="sm:hidden">Create Sandbox</span>
                </button>
                <button onClick={deleteVolume} className="px-3 py-1.5 text-sm bg-red-900 hover:bg-red-800 text-red-200 border border-red-800 rounded-lg transition-colors cursor-pointer">Delete</button>
              </div>
            </header>

            <>
                <div className="px-3 md:px-4 py-2 border-b border-gray-800 text-sm shrink-0 overflow-x-auto">
                  <div className="flex items-center gap-1 whitespace-nowrap">
                    <button onClick={() => navigateVolume('/')} className="text-purple-400 hover:text-purple-300 cursor-pointer">/</button>
                    {volumePath !== '/' && volumePath.split('/').filter(Boolean).map((segment, i, arr) => {
                      const fullPath = '/' + arr.slice(0, i + 1).join('/')
                      return (
                        <span key={fullPath} className="flex items-center gap-1">
                          <span className="text-gray-600">/</span>
                          <button onClick={() => navigateVolume(fullPath)} className="text-purple-400 hover:text-purple-300 cursor-pointer">{segment}</button>
                        </span>
                      )
                    })}
                  </div>
                </div>
                <div className="flex-1 flex min-h-0">
                  <div className={`overflow-y-auto shrink-0 ${viewingFile ? 'hidden md:block md:w-64 md:border-r md:border-gray-800' : 'flex-1'}`}>
                    {volumeLoading ? (
                      <div className="flex items-center justify-center p-8">
                        <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-700 border-t-purple-500"></div>
                      </div>
                    ) : (
                      <div className="divide-y divide-gray-800">
                        {volumePath !== '/' && (
                          <button onClick={() => { const parent = volumePath.split('/').slice(0, -1).join('/') || '/'; navigateVolume(parent) }} className="w-full text-left px-4 py-2 hover:bg-gray-900 flex items-center gap-3 cursor-pointer">
                            <span className="text-gray-500">..</span>
                          </button>
                        )}
                        {volumeEntries.map((entry) => {
                          const name = entry.path.split('/').pop() || entry.path
                          const isDir = entry.type === 'directory'
                          return (
                            <button key={entry.path} onClick={() => isDir ? navigateVolume(entry.path) : openFile(entry.path)}
                              className={`w-full text-left px-4 py-2 hover:bg-gray-900 flex items-center gap-3 cursor-pointer ${viewingFile?.path === entry.path ? 'bg-gray-800' : ''}`}>
                              <span className={`text-sm ${isDir ? 'text-blue-400' : 'text-gray-400'}`}>{isDir ? '\u{1F4C1}' : '\u{1F4C4}'}</span>
                              <span className={`text-sm font-mono ${isDir ? 'text-gray-200' : 'text-gray-400'}`}>{name}</span>
                            </button>
                          )
                        })}
                        {volumeEntries.length === 0 && !volumeLoading && (
                          <p className="text-sm text-gray-500 px-4 py-4">Empty directory</p>
                        )}
                      </div>
                    )}
                  </div>
                  {fileLoading && (
                    <div className="flex-1 flex items-center justify-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-700 border-t-purple-500"></div>
                    </div>
                  )}
                  {viewingFile && !fileLoading && (
                    <div className="flex-1 flex flex-col min-h-0 min-w-0">
                      <div className="flex items-center justify-between px-3 md:px-4 py-2 bg-gray-900 border-b border-gray-800 shrink-0 gap-2">
                        <span className="text-xs font-mono text-gray-400 truncate">{viewingFile.path}</span>
                        <button onClick={() => setViewingFile(null)} className="text-xs text-gray-400 hover:text-gray-200 cursor-pointer shrink-0 px-2 py-1 rounded hover:bg-gray-800">Close</button>
                      </div>
                      {viewingFile.encoding === 'base64' ? (
                        <pre className="flex-1 overflow-auto p-4 text-xs font-mono text-gray-300 bg-gray-950 whitespace-pre-wrap">{`[Binary file — ${viewingFile.content.length} bytes base64]`}</pre>
                      ) : viewingFile.path.endsWith('.ipynb') ? (
                        <div className="flex-1 overflow-auto bg-gray-950"><NotebookViewer content={viewingFile.content} /></div>
                      ) : (
                        <div className="flex-1 overflow-auto">
                          <SyntaxHighlighter
                            language={(() => {
                              const ext = viewingFile.path.split('.').pop()?.toLowerCase() || ''
                              const map: Record<string, string> = {
                                py: 'python', js: 'javascript', ts: 'typescript', tsx: 'tsx', jsx: 'jsx',
                                rs: 'rust', go: 'go', rb: 'ruby', sh: 'bash', bash: 'bash', zsh: 'bash',
                                yaml: 'yaml', yml: 'yaml', json: 'json', toml: 'toml', md: 'markdown',
                                css: 'css', html: 'html', sql: 'sql', c: 'c', cpp: 'cpp', h: 'c',
                                java: 'java', kt: 'kotlin', swift: 'swift', dockerfile: 'docker',
                              }
                              return map[ext] || 'text'
                            })()}
                            style={vscDarkPlus}
                            showLineNumbers
                            lineNumberStyle={{ color: '#4b5563', fontSize: '11px', minWidth: '2.5em' }}
                            customStyle={{ margin: 0, padding: '16px', fontSize: '12px', lineHeight: '1.5' }}
                          >
                            {viewingFile.content}
                          </SyntaxHighlighter>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
          </div>
        ) : (
          <>
            <header className="md:hidden flex items-center gap-3 px-3 py-2 bg-gray-900 border-b border-gray-800 shrink-0">
              <MobileMenuButton onClick={() => setMobileSidebarOpen(true)} />
              <span className="text-sm text-gray-400">Dashboard</span>
            </header>
            <div className="flex-1 flex items-center justify-center text-gray-500 px-4 text-center">
              <p>
                <span className="md:hidden">Open the menu to select a sandbox or volume</span>
                <span className="hidden md:inline">Select a sandbox or volume</span>
              </p>
            </div>
          </>
        )}
      </div>

      {/* Create form modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/60 z-50 overflow-y-auto p-4">
          <div className="min-h-full flex items-center justify-center">
            <div className="w-full max-w-md p-5 sm:p-6 bg-gray-900 rounded-xl border border-gray-800">
              <h2 className="text-xl font-semibold mb-5">Create Sandbox</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">GitHub Repo</label>
                  <input type="text" placeholder="owner/repo" value={githubRepo} onChange={(e) => setGithubRepo(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500" />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Branch / Ref</label>
                  <input type="text" value={githubRef} onChange={(e) => setGithubRef(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500" />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Volume</label>
                  <select value={volumeName} onChange={(e) => setVolumeName(e.target.value)} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500">
                    <option value="">Create new volume</option>
                    {volumes.map((v) => (
                      <option key={v.volume_id} value={v.name}>{v.name}{v.created_at ? ` (${new Date(v.created_at).toLocaleDateString()})` : ''}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">GPU</label>
                  <div className="flex gap-2">
                    <select value={gpu} onChange={(e) => { setGpu(e.target.value); if (!e.target.value) setGpuCount(1) }} className="flex-1 min-w-0 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500">
                      <option value="T4">T4</option>
                      <option value="L4">L4</option>
                      <option value="A10G">A10G</option>
                      <option value="L40S">L40S</option>
                      <option value="A100">A100 (40GB)</option>
                      <option value="A100-80GB">A100 (80GB)</option>
                      <option value="H100">H100</option>
                      <option value="">None (CPU only)</option>
                    </select>
                    <select value={gpuCount} onChange={(e) => setGpuCount(Number(e.target.value))} disabled={!gpu} className="w-16 shrink-0 px-2 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500 disabled:opacity-40">
                      <option value={1}>1x</option>
                      <option value={2}>2x</option>
                      <option value={4}>4x</option>
                      <option value={8}>8x</option>
                    </select>
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">CPU Cores</label>
                  <select value={cpu} onChange={(e) => setCpu(Number(e.target.value))} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500">
                    <option value={2}>2 cores</option>
                    <option value={4}>4 cores</option>
                    <option value={8}>8 cores</option>
                    <option value={16}>16 cores</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Memory</label>
                  <select value={memory} onChange={(e) => setMemory(Number(e.target.value))} className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500">
                    <option value={8192}>8 GiB</option>
                    <option value={16384}>16 GiB</option>
                    <option value={32768}>32 GiB</option>
                    <option value={65536}>64 GiB</option>
                    <option value={131072}>128 GiB</option>
                  </select>
                </div>
              </div>
              <div className="flex gap-3 mt-6">
                <button onClick={() => setShowCreateForm(false)} className="flex-1 py-2.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg font-medium transition-colors cursor-pointer">Cancel</button>
                <button onClick={spawnSession} disabled={spawning} className="flex-1 py-2.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-medium transition-colors cursor-pointer">{spawning ? 'Creating...' : 'Create'}</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Floating action buttons — text chat + voice */}
      {!chatOpen && !voiceOpen && (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3">
          <button
            onClick={() => setVoiceOpen(true)}
            className="w-12 h-12 bg-gray-700 hover:bg-gray-600 rounded-full shadow-lg flex items-center justify-center transition-colors cursor-pointer"
            title="Voice chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-gray-200">
              <path d="M7 4a3 3 0 0 1 6 0v6a3 3 0 1 1-6 0V4Z" />
              <path d="M5.5 9.643a.75.75 0 0 0-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 0 0 0 1.5h4.5a.75.75 0 0 0 0-1.5h-1.5v-1.546A6.001 6.001 0 0 0 16 10v-.357a.75.75 0 0 0-1.5 0V10a4.5 4.5 0 0 1-9 0v-.357Z" />
            </svg>
          </button>
          <button
            onClick={() => setChatOpen(true)}
            className="w-14 h-14 bg-purple-600 hover:bg-purple-500 rounded-full shadow-lg flex items-center justify-center transition-colors cursor-pointer"
            title="Text chat"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-white">
              <path fillRule="evenodd" d="M4.804 21.644A6.707 6.707 0 0 0 6 21.75a6.721 6.721 0 0 0 3.583-1.029c.774.182 1.584.279 2.417.279 5.322 0 9.75-3.97 9.75-9 0-5.03-4.428-9-9.75-9s-9.75 3.97-9.75 9c0 2.409 1.025 4.587 2.674 6.192.232.226.277.428.254.543a3.73 3.73 0 0 1-.814 1.686.75.75 0 0 0 .44 1.223Z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      )}

      {/* Voice chat panel */}
      {voiceOpen && <VoiceChat wsUrl={VOICE_WS_URL} onClose={() => setVoiceOpen(false)} />}

      {/* Floating chat panel */}
      {chatOpen && (
        <div className="fixed bottom-6 right-6 z-50 w-96 max-w-[calc(100vw-3rem)] h-[32rem] max-h-[calc(100dvh-3rem)] flex flex-col bg-gray-900 border border-gray-700 rounded-xl shadow-2xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 shrink-0">
            <span className="text-sm font-medium text-gray-200">Chat</span>
            <div className="flex items-center gap-1 shrink-0">
              <button onClick={() => navigate('/agent/chat')} className="p-1 text-gray-400 hover:text-gray-200 cursor-pointer" title="Open full page">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                  <path fillRule="evenodd" d="M4.25 5.5a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h8.5a.75.75 0 0 0 .75-.75v-4a.75.75 0 0 1 1.5 0v4A2.25 2.25 0 0 1 12.75 17h-8.5A2.25 2.25 0 0 1 2 14.75v-8.5A2.25 2.25 0 0 1 4.25 4h5a.75.75 0 0 1 0 1.5h-5Z" clipRule="evenodd" />
                  <path fillRule="evenodd" d="M6.194 12.753a.75.75 0 0 0 1.06.053L16.5 4.44v2.81a.75.75 0 0 0 1.5 0v-4.5a.75.75 0 0 0-.75-.75h-4.5a.75.75 0 0 0 0 1.5h2.553l-9.056 8.194a.75.75 0 0 0-.053 1.06Z" clipRule="evenodd" />
                </svg>
              </button>
              <button onClick={startNewChat} className="p-1 text-gray-400 hover:text-gray-200 cursor-pointer" title="New chat">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                  <path d="M10.75 4.75a.75.75 0 0 0-1.5 0v4.5h-4.5a.75.75 0 0 0 0 1.5h4.5v4.5a.75.75 0 0 0 1.5 0v-4.5h4.5a.75.75 0 0 0 0-1.5h-4.5v-4.5Z" />
                </svg>
              </button>
              <button onClick={() => setChatOpen(false)} className="p-1 text-gray-400 hover:text-gray-200 cursor-pointer" title="Close">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                  <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
                </svg>
              </button>
            </div>
          </div>

          <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
            {chatMessages.length === 0 && !chatSending && (
              <div className="text-sm text-gray-500 text-center mt-8 space-y-2">
                <p>Ask about any volume — the agent will discover and browse them.</p>
                <p className="text-xs">Uses list_volumes, grep, read_file, read_notebook.</p>
              </div>
            )}
            {chatMessages.map((m) => (
              <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] px-3 py-2 rounded-lg text-sm ${
                  m.role === 'user' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-100 border border-gray-700'
                }`}>
                  {m.parts.map((part, j) => {
                    if (part.type === 'text') {
                      return (
                        <div key={j} className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-pre:my-2 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-code:text-purple-300 prose-a:text-purple-400">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{part.text}</ReactMarkdown>
                        </div>
                      )
                    }
                    if (part.type === 'reasoning') {
                      return (
                        <div key={j} className="text-xs text-gray-500 italic border-l-2 border-gray-700 pl-2 my-1">
                          {part.text}
                        </div>
                      )
                    }
                    if (part.type.startsWith('tool-')) {
                      const toolPart = part as any
                      return (
                        <div key={j} className="text-xs text-gray-400 font-mono bg-gray-900 rounded px-2 py-1 my-1">
                          <span className="text-blue-400">{toolPart.toolName || 'tool'}</span>
                          {toolPart.state === 'result' || toolPart.state === 'output-available' ? (
                            <span className="text-green-400 ml-1">done</span>
                          ) : (
                            <span className="text-yellow-400 ml-1 animate-pulse">running...</span>
                          )}
                        </div>
                      )
                    }
                    return null
                  })}
                </div>
              </div>
            ))}
            {chatStatus === 'submitted' && (
              <div className="flex justify-start">
                <div className="flex items-center gap-2 px-3 py-1.5 text-xs text-gray-400">
                  <div className="w-3 h-3 border-2 border-gray-600 border-t-purple-400 rounded-full animate-spin" />
                  thinking...
                </div>
              </div>
            )}
          </div>

          <div className="border-t border-gray-800 p-3 shrink-0">
            <div className="flex gap-2">
              <textarea
                value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendChat() } }}
                placeholder="What's the latest train loss? Which volumes have results?"
                rows={2} disabled={chatSending}
                className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm resize-none focus:outline-none focus:border-purple-500 disabled:opacity-60"
              />
              <button
                onClick={() => handleSendChat()}
                disabled={chatSending || !chatInput.trim()}
                className="px-3 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-medium transition-colors cursor-pointer self-end"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
