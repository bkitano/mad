import { useCallback, useEffect, useRef, useState } from 'react'

const MODAL_BASE = 'https://miravoice--mad-sandbox-worker'
const CREATE_URL = `${MODAL_BASE}-create-sandbox-worker.modal.run`
const TERMINATE_URL = `${MODAL_BASE}-terminate-sandbox.modal.run`
const LIST_URL = `${MODAL_BASE}-list-sandboxes.modal.run`

interface Session {
  sandbox_id: string
  volume_name: string
  opencode_url: string
  jupyter_url: string
  status: string
}

interface SandboxListItem {
  sandbox_id: string
  opencode_url: string | null
  jupyter_url: string | null
}

function App() {
  const [sandboxes, setSandboxes] = useState<SandboxListItem[]>([])
  const [session, setSession] = useState<Session | null>(null)
  const [spawning, setSpawning] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [activeTab, setActiveTab] = useState<'opencode' | 'jupyter'>('opencode')
  const [jupyterReady, setJupyterReady] = useState(false)
  const jupyterPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Poll jupyter URL until it responds
  const pollJupyter = useCallback((url: string) => {
    setJupyterReady(false)
    if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
    jupyterPollRef.current = setInterval(async () => {
      try {
        const res = await fetch(url, { method: 'HEAD', mode: 'no-cors' })
        // no-cors gives opaque response (status 0) but means the server responded
        if (res.type === 'opaque' || res.ok) {
          setJupyterReady(true)
          if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
        }
      } catch {
        // not ready yet
      }
    }, 3000)
  }, [])

  // Clean up polling on unmount or session change
  useEffect(() => {
    return () => {
      if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
    }
  }, [])

  // Start polling when session changes
  useEffect(() => {
    if (session?.jupyter_url) {
      pollJupyter(session.jupyter_url)
    } else {
      setJupyterReady(false)
      if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
    }
  }, [session?.jupyter_url, pollJupyter])

  // Create form state
  const [githubRepo, setGithubRepo] = useState('bkitano/mad-experiments-template')
  const [githubRef, setGithubRef] = useState('main')
  const [volumeName, setVolumeName] = useState('')
  const [gpu, setGpu] = useState('T4')

  const fetchSandboxes = async () => {
    try {
      const res = await fetch(LIST_URL)
      if (res.ok) {
        const data = await res.json()
        setSandboxes(data.sandboxes || [])
      }
    } catch {
      // silently fail
    }
  }

  useEffect(() => {
    fetchSandboxes()
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
      if (gpu) body.gpu = gpu

      const res = await fetch(CREATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(await res.text())
      const data: Session = await res.json()
      setSession(data)
      setShowCreateForm(false)
      fetchSandboxes()
    } catch (err) {
      alert(`Failed to spawn session: ${err}`)
    } finally {
      setSpawning(false)
    }
  }

  const selectSandbox = (sb: SandboxListItem) => {
    if (!sb.opencode_url) return
    setSession({
      sandbox_id: sb.sandbox_id,
      volume_name: '',
      opencode_url: sb.opencode_url,
      jupyter_url: sb.jupyter_url || '',
      status: 'running',
    })
  }

  const terminateSession = async () => {
    if (!session) return
    if (!confirm('Terminate this session?')) return
    try {
      await fetch(TERMINATE_URL, {
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

  return (
    <div className="h-screen flex bg-gray-950 text-gray-100">
      {/* Sidebar */}
      <aside className="w-64 flex flex-col border-r border-gray-800 bg-gray-900">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-semibold">Sandboxes</h1>
        </div>

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
            onClick={() => setShowCreateForm(true)}
            className="w-full py-2 bg-purple-600 hover:bg-purple-500 rounded-lg text-sm font-medium transition-colors cursor-pointer"
          >
            Create Sandbox
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {session ? (
          <>
            {/* Top bar */}
            <header className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-800">
              <div className="flex items-center gap-4">
                <span className="text-sm font-mono text-gray-400">
                  {session.sandbox_id.slice(0, 12)}
                </span>
                <span className="text-xs px-2 py-0.5 bg-green-900 text-green-300 rounded">
                  {session.status}
                </span>
                {session.volume_name && (
                  <span className="text-xs text-gray-500">vol: {session.volume_name}</span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <div className="flex bg-gray-800 rounded-lg p-0.5 mr-4">
                  <button
                    onClick={() => setActiveTab('opencode')}
                    className={`px-3 py-1 text-sm rounded-md transition-colors cursor-pointer ${
                      activeTab === 'opencode'
                        ? 'bg-purple-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    OpenCode
                  </button>
                  <button
                    onClick={() => setActiveTab('jupyter')}
                    className={`px-3 py-1 text-sm rounded-md transition-colors cursor-pointer ${
                      activeTab === 'jupyter'
                        ? 'bg-purple-600 text-white'
                        : 'text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    Jupyter
                  </button>
                </div>

                <button
                  onClick={exportFiles}
                  className="px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg transition-colors cursor-pointer"
                >
                  Export Files
                </button>
                <button
                  onClick={terminateSession}
                  className="px-3 py-1.5 text-sm bg-red-900 hover:bg-red-800 text-red-200 border border-red-800 rounded-lg transition-colors cursor-pointer"
                >
                  Terminate
                </button>
              </div>
            </header>

            {/* Embedded views */}
            <main className="flex-1 relative">
              <iframe
                src={session.opencode_url}
                className={`absolute inset-0 w-full h-full border-0 ${
                  activeTab === 'opencode' ? 'block' : 'hidden'
                }`}
                title="OpenCode"
              />
              {activeTab === 'jupyter' && !jupyterReady && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-950 z-10">
                  <div className="animate-spin rounded-full h-10 w-10 border-2 border-gray-700 border-t-purple-500 mb-4"></div>
                  <p className="text-sm text-gray-400">Waiting for Jupyter to start...</p>
                </div>
              )}
              <iframe
                src={jupyterReady ? session.jupyter_url : undefined}
                className={`absolute inset-0 w-full h-full border-0 ${
                  activeTab === 'jupyter' ? 'block' : 'hidden'
                }`}
                title="Jupyter"
              />
            </main>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            <p>Select a sandbox or create a new one</p>
          </div>
        )}
      </div>

      {/* Create form modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="w-full max-w-md p-6 bg-gray-900 rounded-xl border border-gray-800">
            <h2 className="text-xl font-semibold mb-5">Create Sandbox</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">GitHub Repo</label>
                <input
                  type="text"
                  placeholder="owner/repo"
                  value={githubRepo}
                  onChange={(e) => setGithubRepo(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">Branch / Ref</label>
                <input
                  type="text"
                  value={githubRef}
                  onChange={(e) => setGithubRef(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">Volume Name (attach existing)</label>
                <input
                  type="text"
                  placeholder="Leave empty for new volume"
                  value={volumeName}
                  onChange={(e) => setVolumeName(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">GPU</label>
                <select
                  value={gpu}
                  onChange={(e) => setGpu(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                >
                  <option value="T4">T4</option>
                  <option value="A10G">A10G</option>
                  <option value="A100">A100</option>
                  <option value="">None</option>
                </select>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setShowCreateForm(false)}
                className="flex-1 py-2.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg font-medium transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={spawnSession}
                disabled={spawning}
                className="flex-1 py-2.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-medium transition-colors cursor-pointer"
              >
                {spawning ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
