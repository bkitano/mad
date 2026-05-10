import { useCallback, useEffect, useRef, useState } from 'react'
import NotebookViewer from './NotebookViewer'

const MODAL_BASE = 'https://miravoice--mad-sandbox-worker'
const CREATE_URL = `${MODAL_BASE}-create-sandbox-worker.modal.run`
const TERMINATE_URL = `${MODAL_BASE}-terminate-sandbox.modal.run`
const LIST_URL = `${MODAL_BASE}-list-sandboxes.modal.run`
const VOLUMES_URL = `${MODAL_BASE}-list-volumes.modal.run`
const VOLUME_LS_URL = `${MODAL_BASE}-volume-ls.modal.run`
const VOLUME_READ_URL = `${MODAL_BASE}-volume-read.modal.run`
const RENAME_VOLUME_URL = `${MODAL_BASE}-rename-volume.modal.run`
const DELETE_VOLUME_URL = `${MODAL_BASE}-delete-volume.modal.run`

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
  volume_name: string
}

interface VolumeItem {
  name: string
  volume_id: string
  created_at: string | null
  created_by: string | null
}

type SidebarTab = 'sandboxes' | 'volumes'

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

function App() {
  const [sandboxes, setSandboxes] = useState<SandboxListItem[]>([])
  const [session, setSession] = useState<Session | null>(null)
  const [spawning, setSpawning] = useState(false)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [activeTab, setActiveTab] = useState<'opencode' | 'jupyter'>('opencode')
  const [jupyterReady, setJupyterReady] = useState(false)
  const jupyterPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Sidebar navigation
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('sandboxes')
  const [volumes, setVolumes] = useState<VolumeItem[]>([])
  const [selectedVolume, setSelectedVolume] = useState<VolumeItem | null>(null)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)

  // Volume file browser
  const [volumePath, setVolumePath] = useState('/')
  const [volumeEntries, setVolumeEntries] = useState<{ path: string; type: string }[]>([])
  const [volumeLoading, setVolumeLoading] = useState(false)
  const [viewingFile, setViewingFile] = useState<{ path: string; content: string; encoding: string } | null>(null)
  const [fileLoading, setFileLoading] = useState(false)

  // Volume rename
  const [renamingVolume, setRenamingVolume] = useState(false)
  const [renameValue, setRenameValue] = useState('')
  const [renameLoading, setRenameLoading] = useState(false)

  // Poll jupyter URL until it responds
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
    return () => {
      if (jupyterPollRef.current) clearInterval(jupyterPollRef.current)
    }
  }, [])

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
  const [gpuCount, setGpuCount] = useState(1)
  const [cpu, setCpu] = useState(4)
  const [memory, setMemory] = useState(32768)

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

  const fetchVolumes = async () => {
    try {
      const res = await fetch(VOLUMES_URL)
      if (res.ok) {
        const data = await res.json()
        setVolumes(data.volumes || [])
      }
    } catch {
      // silently fail
    }
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

      const res = await fetch(CREATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(await res.text())
      const data: Session = await res.json()
      setSession(data)
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
    })
    setMobileSidebarOpen(false)
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

  const fetchVolumeContents = async (volName: string, path: string) => {
    setVolumeLoading(true)
    try {
      const params = new URLSearchParams({ volume_name: volName, path })
      const res = await fetch(`${VOLUME_LS_URL}?${params}`)
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
      const res = await fetch(`${VOLUME_READ_URL}?${params}`)
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
      const res = await fetch(RENAME_VOLUME_URL, {
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
      const res = await fetch(DELETE_VOLUME_URL, {
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
      {/* Mobile sidebar backdrop — blocks touch events from reaching the iframe behind */}
      {mobileSidebarOpen && (
        <div
          onClick={() => setMobileSidebarOpen(false)}
          className="md:hidden fixed inset-0 bg-black/60 z-30"
          aria-hidden="true"
        />
      )}

      {/* Sidebar — overlay drawer on mobile, static column on md+ */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-64 flex flex-col border-r border-gray-800 bg-gray-900 transform transition-transform duration-200 md:translate-x-0 ${
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Sidebar tab switcher */}
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
        </div>

        {/* Sidebar content */}
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
      </aside>

      {/* Main content — min-w-0 prevents iframe from forcing the column wider than viewport */}
      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        {session ? (
          <>
            {/* Top bar */}
            <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between px-3 py-2 md:px-4 bg-gray-900 border-b border-gray-800 shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <MobileMenuButton onClick={() => setMobileSidebarOpen(true)} />
                <span className="text-sm font-mono text-gray-400 truncate">
                  {session.sandbox_id.slice(0, 12)}
                </span>
                <span className="text-xs px-2 py-0.5 bg-green-900 text-green-300 rounded shrink-0">
                  {session.status}
                </span>
                {session.volume_name && (
                  <span className="hidden md:inline text-xs text-gray-500 truncate">
                    vol: {session.volume_name}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex bg-gray-800 rounded-lg p-0.5">
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
                  Export
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
        ) : selectedVolume ? (
          /* Volume detail + file browser */
          <div className="flex-1 flex flex-col min-h-0">
            {/* Volume header */}
            <header className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between px-3 py-2 md:px-4 bg-gray-900 border-b border-gray-800 shrink-0">
              <div className="flex items-center gap-3 min-w-0">
                <MobileMenuButton onClick={() => setMobileSidebarOpen(true)} />
                {renamingVolume ? (
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <input
                      type="text"
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') submitRename()
                        if (e.key === 'Escape') setRenamingVolume(false)
                      }}
                      autoFocus
                      disabled={renameLoading}
                      className="flex-1 min-w-0 sm:w-64 sm:flex-none px-2 py-1 bg-gray-800 border border-gray-600 rounded text-sm font-mono text-gray-200 focus:outline-none focus:border-purple-500"
                    />
                    <button
                      onClick={submitRename}
                      disabled={renameLoading}
                      className="px-2 py-1 text-xs bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 rounded transition-colors cursor-pointer shrink-0"
                    >
                      {renameLoading ? '...' : 'Save'}
                    </button>
                    <button
                      onClick={() => setRenamingVolume(false)}
                      disabled={renameLoading}
                      className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200 cursor-pointer shrink-0"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <>
                    <span className="text-sm font-mono text-gray-300 truncate">{selectedVolume.name}</span>
                    <button
                      onClick={startRename}
                      className="text-gray-500 hover:text-gray-300 cursor-pointer shrink-0"
                      title="Rename volume"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                        <path d="M2.695 14.763l-1.262 3.154a.5.5 0 00.65.65l3.155-1.262a4 4 0 001.343-.885L17.5 5.5a2.121 2.121 0 00-3-3L3.58 13.42a4 4 0 00-.885 1.343z" />
                      </svg>
                    </button>
                    <span className="hidden md:inline text-xs text-gray-500 shrink-0">
                      {selectedVolume.created_at
                        ? new Date(selectedVolume.created_at).toLocaleString()
                        : ''}
                    </span>
                  </>
                )}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={() => handleAttachAndCreate(selectedVolume)}
                  className="px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-500 rounded-lg font-medium transition-colors cursor-pointer"
                >
                  <span className="hidden sm:inline">Create Sandbox with this Volume</span>
                  <span className="sm:hidden">Create Sandbox</span>
                </button>
                <button
                  onClick={deleteVolume}
                  className="px-3 py-1.5 text-sm bg-red-900 hover:bg-red-800 text-red-200 border border-red-800 rounded-lg transition-colors cursor-pointer"
                >
                  Delete
                </button>
              </div>
            </header>

            {/* Breadcrumb — horizontal scroll instead of wrap so deep paths stay readable */}
            <div className="px-3 md:px-4 py-2 border-b border-gray-800 text-sm shrink-0 overflow-x-auto">
              <div className="flex items-center gap-1 whitespace-nowrap">
                <button
                  onClick={() => navigateVolume('/')}
                  className="text-purple-400 hover:text-purple-300 cursor-pointer"
                >
                  /
                </button>
                {volumePath !== '/' && volumePath.split('/').filter(Boolean).map((segment, i, arr) => {
                  const fullPath = '/' + arr.slice(0, i + 1).join('/')
                  return (
                    <span key={fullPath} className="flex items-center gap-1">
                      <span className="text-gray-600">/</span>
                      <button
                        onClick={() => navigateVolume(fullPath)}
                        className="text-purple-400 hover:text-purple-300 cursor-pointer"
                      >
                        {segment}
                      </button>
                    </span>
                  )
                })}
              </div>
            </div>

            {/* File listing + content. On mobile, listing hides when a file is open;
                "Close" in the file viewer brings the listing back. */}
            <div className="flex-1 flex min-h-0">
              <div
                className={`overflow-y-auto shrink-0 ${
                  viewingFile
                    ? 'hidden md:block md:w-64 md:border-r md:border-gray-800'
                    : 'flex-1'
                }`}
              >
                {volumeLoading ? (
                  <div className="flex items-center justify-center p-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-700 border-t-purple-500"></div>
                  </div>
                ) : (
                  <div className="divide-y divide-gray-800">
                    {volumePath !== '/' && (
                      <button
                        onClick={() => {
                          const parent = volumePath.split('/').slice(0, -1).join('/') || '/'
                          navigateVolume(parent)
                        }}
                        className="w-full text-left px-4 py-2 hover:bg-gray-900 flex items-center gap-3 cursor-pointer"
                      >
                        <span className="text-gray-500">..</span>
                      </button>
                    )}
                    {volumeEntries.map((entry) => {
                      const name = entry.path.split('/').pop() || entry.path
                      const isDir = entry.type === 'directory'
                      return (
                        <button
                          key={entry.path}
                          onClick={() => isDir ? navigateVolume(entry.path) : openFile(entry.path)}
                          className={`w-full text-left px-4 py-2 hover:bg-gray-900 flex items-center gap-3 cursor-pointer ${
                            viewingFile?.path === entry.path ? 'bg-gray-800' : ''
                          }`}
                        >
                          <span className={`text-sm ${isDir ? 'text-blue-400' : 'text-gray-400'}`}>
                            {isDir ? '📁' : '📄'}
                          </span>
                          <span className={`text-sm font-mono ${isDir ? 'text-gray-200' : 'text-gray-400'}`}>
                            {name}
                          </span>
                        </button>
                      )
                    })}
                    {volumeEntries.length === 0 && !volumeLoading && (
                      <p className="text-sm text-gray-500 px-4 py-4">Empty directory</p>
                    )}
                  </div>
                )}
              </div>

              {/* File content viewer */}
              {fileLoading && (
                <div className="flex-1 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-700 border-t-purple-500"></div>
                </div>
              )}
              {viewingFile && !fileLoading && (
                <div className="flex-1 flex flex-col min-h-0 min-w-0">
                  <div className="flex items-center justify-between px-3 md:px-4 py-2 bg-gray-900 border-b border-gray-800 shrink-0 gap-2">
                    <span className="text-xs font-mono text-gray-400 truncate">{viewingFile.path}</span>
                    <button
                      onClick={() => setViewingFile(null)}
                      className="text-xs text-gray-400 hover:text-gray-200 cursor-pointer shrink-0 px-2 py-1 rounded hover:bg-gray-800"
                    >
                      Close
                    </button>
                  </div>
                  {viewingFile.encoding === 'base64' ? (
                    <pre className="flex-1 overflow-auto p-4 text-xs font-mono text-gray-300 bg-gray-950 whitespace-pre-wrap">
                      {`[Binary file — ${viewingFile.content.length} bytes base64]`}
                    </pre>
                  ) : viewingFile.path.endsWith('.ipynb') ? (
                    <div className="flex-1 overflow-auto bg-gray-950">
                      <NotebookViewer content={viewingFile.content} />
                    </div>
                  ) : (
                    <pre className="flex-1 overflow-auto p-4 text-xs font-mono text-gray-300 bg-gray-950 whitespace-pre-wrap">
                      {viewingFile.content}
                    </pre>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          <>
            {/* Mobile-only header so the sidebar is reachable from the empty state */}
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

      {/* Create form modal — outer scrolls so the form is reachable on short screens */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black/60 z-50 overflow-y-auto p-4">
          <div className="min-h-full flex items-center justify-center">
            <div className="w-full max-w-md p-5 sm:p-6 bg-gray-900 rounded-xl border border-gray-800">
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
                  <label className="block text-sm text-gray-400 mb-1">Volume</label>
                  <select
                    value={volumeName}
                    onChange={(e) => setVolumeName(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                  >
                    <option value="">Create new volume</option>
                    {volumes.map((v) => (
                      <option key={v.volume_id} value={v.name}>
                        {v.name}{v.created_at ? ` (${new Date(v.created_at).toLocaleDateString()})` : ''}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">GPU</label>
                  <div className="flex gap-2">
                    <select
                      value={gpu}
                      onChange={(e) => { setGpu(e.target.value); if (!e.target.value) setGpuCount(1) }}
                      className="flex-1 min-w-0 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                    >
                      <option value="T4">T4</option>
                      <option value="L4">L4</option>
                      <option value="A10G">A10G</option>
                      <option value="L40S">L40S</option>
                      <option value="A100">A100 (40GB)</option>
                      <option value="A100-80GB">A100 (80GB)</option>
                      <option value="H100">H100</option>
                      <option value="">None (CPU only)</option>
                    </select>
                    <select
                      value={gpuCount}
                      onChange={(e) => setGpuCount(Number(e.target.value))}
                      disabled={!gpu}
                      className="w-16 shrink-0 px-2 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500 disabled:opacity-40"
                    >
                      <option value={1}>1x</option>
                      <option value={2}>2x</option>
                      <option value={4}>4x</option>
                      <option value={8}>8x</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">CPU Cores</label>
                  <select
                    value={cpu}
                    onChange={(e) => setCpu(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                  >
                    <option value={2}>2 cores</option>
                    <option value={4}>4 cores</option>
                    <option value={8}>8 cores</option>
                    <option value={16}>16 cores</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">Memory</label>
                  <select
                    value={memory}
                    onChange={(e) => setMemory(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-purple-500"
                  >
                    <option value={8192}>8 GiB</option>
                    <option value={16384}>16 GiB</option>
                    <option value={32768}>32 GiB</option>
                    <option value={65536}>64 GiB</option>
                    <option value={131072}>128 GiB</option>
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
        </div>
      )}
    </div>
  )
}

export default App
