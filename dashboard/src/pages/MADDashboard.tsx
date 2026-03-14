import { useState, useEffect, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import ExperimentCard from '../components/mad/ExperimentCard'
import LogViewer from '../components/mad/LogViewer'
import ProposalsView from '../components/mad/ProposalsView'
import TricksView from '../components/mad/TricksView'

import CodeViewer from '../components/mad/CodeViewer'
import WorkersView from '../components/mad/WorkersView'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface ActiveExperiment {
  proposal_id: string
  started_at: string
  status: string
}

interface HistoryItem extends ActiveExperiment {
  completed_at?: string
  staled_at?: string
}

interface DashboardData {
  active_work: Record<string, ActiveExperiment>
  history: HistoryItem[]
  server_timestamp?: string
}

interface Experiment {
  id: string
  proposal_id: string
  status: string
  worker_id?: string
  wandb_url?: string
  results?: Record<string, unknown>
  error?: string
  error_class?: string
  cost_actual?: number
  cost_estimate?: number
  created_at: string
  completed_at?: string
  submitted_at?: string
  artifacts_url?: string
}

type TabType = 'experiments' | 'proposals' | 'tricks' | 'workers'

const API_URL = import.meta.env.VITE_API_URL || 'https://mad.briankitano.com'

export default function MADDashboard() {
  const navigate = useNavigate()
  const location = useLocation()

  // Determine active tab from path
  const getActiveTab = (): TabType => {
    const path = location.pathname
    if (path.includes('/proposals')) return 'proposals'
    if (path.includes('/tricks')) return 'tricks'

    if (path.includes('/workers')) return 'workers'
    return 'experiments'
  }
  const activeTab = getActiveTab()

  const [data, setData] = useState<DashboardData | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [selectedResult, setSelectedResult] = useState<{ id: string; content: string } | null>(null)
  const [selectedLog, setSelectedLog] = useState<{ id: string; events: Array<{ created_at: string; type: string; summary: string; worker_id?: string }> } | null>(null)
  const selectedLogRef = useRef(selectedLog)
  selectedLogRef.current = selectedLog
  const logContainerRef = useRef<HTMLDivElement>(null)
  const [selectedCode, setSelectedCode] = useState<string | null>(null)
  const [selectedArtifacts, setSelectedArtifacts] = useState<{
    id: string
    proposal_id: string
    artifacts_url?: string
    code_files: string[]
    wandb_url?: string
    results?: Record<string, unknown>
  } | null>(null)
  const [isInitialLoad, setIsInitialLoad] = useState(true)

  // All experiments list
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [experimentsLoading, setExperimentsLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [experimentsPage, setExperimentsPage] = useState(0)
  const EXPERIMENTS_PER_PAGE = 25

  // Fetch all experiments
  useEffect(() => {
    const fetchExperiments = async () => {
      setExperimentsLoading(true)
      try {
        const params = new URLSearchParams({
          limit: String(EXPERIMENTS_PER_PAGE),
          offset: String(experimentsPage * EXPERIMENTS_PER_PAGE),
        })
        if (statusFilter) params.set('status', statusFilter)
        const res = await fetch(`${API_URL}/experiments?${params}`)
        if (res.ok) {
          setExperiments(await res.json())
        }
      } catch (err) {
        console.error('Error fetching experiments:', err)
      } finally {
        setExperimentsLoading(false)
      }
    }
    fetchExperiments()
  }, [statusFilter, experimentsPage])

  // Fetch initial state immediately
  useEffect(() => {
    const fetchInitialState = async () => {
      try {
        console.log('Fetching initial state...')
        const [runningRes, statsRes] = await Promise.all([
          fetch(`${API_URL}/experiments?status=running`),
          fetch(`${API_URL}/stats`)
        ])

        if (runningRes.ok && statsRes.ok) {
          const runningExps = await runningRes.json()
          await statsRes.json()

          const activeWork: Record<string, ActiveExperiment> = {}
          for (const exp of runningExps) {
            activeWork[exp.proposal_id] = {
              proposal_id: exp.proposal_id,
              started_at: exp.created_at,
              status: exp.status
            }
          }

          setData({ active_work: activeWork, history: [] })
          setLastUpdate(new Date())
          setIsInitialLoad(false)
          console.log('Initial state loaded')
        }
      } catch (err) {
        console.error('Error fetching initial state:', err)
      }
    }

    fetchInitialState()
  }, [])

  // SSE connection for live updates
  useEffect(() => {
    let eventSource: EventSource | null = null
    let reconnectTimeout: NodeJS.Timeout

    const connect = () => {
      console.log('Connecting to SSE stream...')
      eventSource = new EventSource(`${API_URL}/events/stream`)

      eventSource.onopen = () => {
        console.log('SSE connected')
        setConnected(true)
        setError(null)
      }

      eventSource.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data)
          console.log('Received event:', event.type)

          // Handle experiment events — refetch running experiments
          if (event.type?.startsWith('experiment.')) {
            fetch(`${API_URL}/experiments?status=running`)
              .then(res => res.json())
              .then(runningExps => {
                const activeWork: Record<string, ActiveExperiment> = {}
                for (const exp of runningExps) {
                  activeWork[exp.proposal_id] = {
                    proposal_id: exp.proposal_id,
                    started_at: exp.created_at,
                    status: exp.status
                  }
                }
                setData(prev => ({ ...prev, active_work: activeWork } as DashboardData))
                setLastUpdate(new Date())
              })
              .catch(err => console.error('Error refetching experiments:', err))
          }
        } catch (err) {
          console.error('Error parsing event data:', err)
        }
      }

      eventSource.onerror = () => {
        console.error('SSE connection error')
        setConnected(false)
        setError('Connection lost. Reconnecting...')
        eventSource?.close()

        // Reconnect after 5 seconds
        reconnectTimeout = setTimeout(connect, 5000)
      }
    }

    connect()

    return () => {
      eventSource?.close()
      clearTimeout(reconnectTimeout)
    }
  }, [])

  // Fetch experiment log (events)
  const fetchLog = async (proposalId: string) => {
    try {
      const experimentId = proposalId.split('-')[0]
      const res = await fetch(`${API_URL}/experiments/${experimentId}/events?limit=200`)
      if (res.ok) {
        const events = await res.json()
        setSelectedLog({ id: proposalId, events })
      } else {
        setSelectedLog({ id: proposalId, events: [] })
      }
    } catch (err) {
      console.error('Error fetching log:', err)
      setSelectedLog({ id: proposalId, events: [] })
    }
  }

  // SSE subscription for live log updates when modal is open
  useEffect(() => {
    if (!selectedLog) return
    const experimentId = selectedLog.id.split('-')[0]
    const es = new EventSource(`${API_URL}/events/stream`)
    es.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data)
        if (String(event.experiment_id) === experimentId) {
          setSelectedLog(prev => prev ? { ...prev, events: [...prev.events, event] } : prev)
          // Auto-scroll
          requestAnimationFrame(() => {
            if (logContainerRef.current) {
              logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
            }
          })
        }
      } catch { /* ignore */ }
    }
    return () => es.close()
  }, [selectedLog?.id])

  // Fetch experiment results
  const fetchResult = async (proposalId: string) => {
    try {
      // Extract experiment number from proposal ID (e.g., "029-some-name" -> "029")
      const experimentId = proposalId.split('-')[0]

      // Check if experiment is currently active
      const isActive = data?.active_work && proposalId in data.active_work

      const res = await fetch(`${API_URL}/experiments/${experimentId}`)
      if (res.ok) {
        const experiment = await res.json()
        if (experiment.results) {
          // Format results as markdown
          const resultsContent = `# Experiment ${experimentId} Results\n\n` +
            `**Status:** ${experiment.status}\n\n` +
            (experiment.wandb_url ? `**W&B:** [View Run](${experiment.wandb_url})\n\n` : '') +
            `## Results\n\n\`\`\`json\n${JSON.stringify(experiment.results, null, 2)}\n\`\`\``
          setSelectedResult({ id: proposalId, content: resultsContent })
        } else {
          const message = isActive
            ? `# Experiment In Progress\n\nExperiment ${experimentId} (${proposalId}) is currently running.\n\nCheck back later for results, or view the Active Experiments section above for real-time status.`
            : `# Results Not Available\n\nNo results found for experiment ${experimentId} (${proposalId}).\n\n**Status:** ${experiment.status}`
          setSelectedResult({ id: proposalId, content: message })
        }
      } else {
        const message = `# Experiment Not Found\n\nExperiment ${experimentId} (${proposalId}) was not found.`
        setSelectedResult({ id: proposalId, content: message })
      }
    } catch (err) {
      console.error('Error fetching result:', err)
      setSelectedResult({ id: proposalId, content: `# Error\n\nFailed to load results: ${err}` })
    }
  }

  // Fetch experiment artifacts
  const fetchArtifacts = async (exp: Experiment) => {
    try {
      const res = await fetch(`${API_URL}/experiments/${exp.id}/artifacts`)
      if (res.ok) {
        setSelectedArtifacts(await res.json())
      } else {
        console.error('Failed to fetch artifacts')
      }
    } catch (err) {
      console.error('Error fetching artifacts:', err)
    }
  }

  // Calculate stats
  const activeExperiments = data ? Object.keys(data.active_work).length : 0
  const totalExperiments = experiments.length
  const completedToday = experiments.filter(exp => {
    if (!exp.completed_at) return false
    const completed = new Date(exp.completed_at)
    const today = new Date()
    return completed.toDateString() === today.toDateString()
  }).length

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">😡 MAD Architecture Search</h1>
        <p className="mt-2 text-gray-600">
          Real-time experiment monitoring dashboard
        </p>
      </div>


      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex gap-8">
          <button
            onClick={() => navigate('/')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'experiments'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Experiments
          </button>
          <button
            onClick={() => navigate('/proposals')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'proposals'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Proposals
          </button>
          <button
            onClick={() => navigate('/tricks')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'tricks'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Tricks
          </button>
          <button
            onClick={() => navigate('/workers')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'workers'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Workers
          </button>
        </nav>
      </div>

      {/* Connection Status (only show on experiments tab) */}
      {activeTab === 'experiments' && (
        <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className={connected ? 'text-green-700' : 'text-red-700'}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        {lastUpdate && (
          <span className="text-gray-500">
            Last update: {lastUpdate.toLocaleTimeString()}
          </span>
        )}
        {error && (
          <span className="text-red-600">{error}</span>
        )}
        </div>
      )}

      {/* Content based on active tab */}
      {activeTab === 'experiments' && (
        <>
          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            {isInitialLoad ? (
              <>
                <div className="bg-blue-50 p-4 rounded-lg animate-pulse">
                  <div className="h-8 w-12 bg-blue-200 rounded mb-2"></div>
                  <div className="h-4 w-32 bg-blue-200 rounded"></div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg animate-pulse">
                  <div className="h-8 w-12 bg-green-200 rounded mb-2"></div>
                  <div className="h-4 w-32 bg-green-200 rounded"></div>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg animate-pulse">
                  <div className="h-8 w-12 bg-gray-300 rounded mb-2"></div>
                  <div className="h-4 w-32 bg-gray-300 rounded"></div>
                </div>
              </>
            ) : (
              <>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-blue-900">{activeExperiments}</div>
                  <div className="text-sm text-blue-700">Active Experiments</div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-green-900">{completedToday}</div>
                  <div className="text-sm text-green-700">Completed Today</div>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-gray-900">{totalExperiments}</div>
                  <div className="text-sm text-gray-700">Experiments (this page)</div>
                </div>
              </>
            )}
          </div>

          {/* Active Experiments */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Active Experiments</h2>
            {isInitialLoad ? (
              <div className="bg-gray-50 p-8 rounded-lg animate-pulse">
                <div className="h-6 bg-gray-300 rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-gray-300 rounded w-1/2 mb-2"></div>
                <div className="h-4 bg-gray-300 rounded w-2/3"></div>
              </div>
            ) : activeExperiments === 0 ? (
              <div className="bg-gray-50 p-8 rounded-lg text-center text-gray-500">
                No experiments currently running
              </div>
            ) : (
              <div className="space-y-4">
                {Object.entries(data?.active_work || {}).map(([proposalId, exp]) => (
                  <ExperimentCard
                    key={proposalId}
                    proposalId={proposalId}
                    experiment={exp}
                    apiUrl={API_URL}
                    onViewProposal={() => {
                      navigate(`/proposals/${proposalId}`)
                    }}
                    onViewLog={fetchLog}
                    onViewCode={setSelectedCode}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Recent Logs */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Recent Logs</h2>
            <LogViewer apiUrl={API_URL} />
          </div>

          {/* All Experiments */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-semibold text-gray-900">All Experiments</h2>
              <div className="flex items-center gap-2">
                <select
                  value={statusFilter}
                  onChange={(e) => { setStatusFilter(e.target.value); setExperimentsPage(0) }}
                  className="text-sm border border-gray-300 rounded-md px-3 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All statuses</option>
                  <option value="created">Created</option>
                  <option value="code_ready">Code Ready</option>
                  <option value="submitted">Submitted</option>
                  <option value="running">Running</option>
                  <option value="completed">Completed</option>
                  <option value="failed">Failed</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>
            </div>
            {experimentsLoading ? (
              <div className="space-y-2">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="p-3 bg-gray-50 rounded-lg animate-pulse">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="h-5 w-16 bg-gray-300 rounded"></div>
                        <div className="h-4 w-48 bg-gray-300 rounded"></div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="h-5 w-20 bg-gray-300 rounded"></div>
                        <div className="h-4 w-32 bg-gray-300 rounded"></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : experiments.length === 0 ? (
              <div className="bg-gray-50 p-8 rounded-lg text-center text-gray-500">
                No experiments found{statusFilter ? ` with status "${statusFilter}"` : ''}
              </div>
            ) : (
              <>
                <div className="overflow-hidden rounded-lg border border-gray-200">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Proposal</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Worker</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Cost</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Links</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {experiments.map((exp) => (
                        <tr
                          key={exp.id}
                          className="hover:bg-gray-50 cursor-pointer"
                          onClick={() => fetchResult(exp.proposal_id)}
                        >
                          <td className="px-4 py-3 text-sm font-mono font-medium text-gray-900">{exp.id}</td>
                          <td className="px-4 py-3 text-sm text-gray-700 max-w-[200px] truncate">{exp.proposal_id}</td>
                          <td className="px-4 py-3">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                              exp.status === 'completed' ? 'bg-green-100 text-green-800' :
                              exp.status === 'failed' ? 'bg-red-100 text-red-800' :
                              exp.status === 'running' ? 'bg-blue-100 text-blue-800' :
                              exp.status === 'cancelled' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {exp.status}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-sm font-mono text-gray-500">
                            {exp.worker_id || '—'}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-500">
                            {new Date(exp.created_at).toLocaleString()}
                          </td>
                          <td className="px-4 py-3 text-sm text-gray-500">
                            {exp.cost_actual != null ? `$${exp.cost_actual.toFixed(2)}` : exp.cost_estimate != null ? `~$${exp.cost_estimate.toFixed(2)}` : '—'}
                          </td>
                          <td className="px-4 py-3 text-sm">
                            <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                              {exp.wandb_url && (
                                <a href={exp.wandb_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 text-xs underline">W&B</a>
                              )}
                              <button
                                onClick={() => setSelectedCode(exp.id)}
                                className="text-blue-600 hover:text-blue-800 text-xs underline"
                              >
                                Code
                              </button>
                              <button
                                onClick={() => fetchLog(exp.proposal_id)}
                                className="text-blue-600 hover:text-blue-800 text-xs underline"
                              >
                                Log
                              </button>
                              <button
                                onClick={() => fetchArtifacts(exp)}
                                className="text-blue-600 hover:text-blue-800 text-xs underline"
                              >
                                Artifacts
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {/* Pagination */}
                <div className="flex items-center justify-between mt-3">
                  <span className="text-sm text-gray-500">
                    Page {experimentsPage + 1} · Showing {experiments.length} experiment{experiments.length !== 1 ? 's' : ''}
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setExperimentsPage(p => Math.max(0, p - 1))}
                      disabled={experimentsPage === 0}
                      className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
                    >
                      Previous
                    </button>
                    <button
                      onClick={() => setExperimentsPage(p => p + 1)}
                      disabled={experiments.length < EXPERIMENTS_PER_PAGE}
                      className="px-3 py-1 text-sm border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Debug Info */}
          {data && (
            <details className="text-xs text-gray-500">
              <summary className="cursor-pointer hover:text-gray-700">Debug Info</summary>
              <pre className="mt-2 p-4 bg-gray-50 rounded overflow-auto max-h-96">
                {JSON.stringify(data, null, 2)}
              </pre>
            </details>
          )}
        </>
      )}

      {activeTab === 'proposals' && <ProposalsView apiUrl={API_URL} />}
      {activeTab === 'tricks' && <TricksView apiUrl={API_URL} />}
      {activeTab === 'workers' && <WorkersView apiUrl={API_URL} />}

      {/* Results Modal */}
      {selectedResult && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedResult(null)}>
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">
                Results: {selectedResult.id}
              </h3>
              <button
                onClick={() => setSelectedResult(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl leading-none"
              >
                ×
              </button>
            </div>
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {selectedResult.content}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Experiment Log Modal */}
      {selectedLog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedLog(null)}>
          <div className="bg-gray-900 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-4 py-3 bg-gray-800">
              <div className="flex items-center gap-3">
                <span className="text-sm font-mono text-gray-300">
                  Logs: {selectedLog.id}
                </span>
                <span className="text-xs text-gray-400">
                  {selectedLog.events.length} events
                </span>
                <span className="inline-flex items-center gap-1 text-xs text-green-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                  Live
                </span>
              </div>
              <button
                onClick={() => setSelectedLog(null)}
                className="text-gray-400 hover:text-gray-200 text-2xl leading-none"
              >
                ×
              </button>
            </div>
            <div ref={logContainerRef} className="p-4 overflow-y-auto max-h-[calc(90vh-60px)] font-mono text-xs">
              {selectedLog.events.length === 0 ? (
                <div className="text-gray-400 text-center py-8">No events recorded for this experiment.</div>
              ) : (
                selectedLog.events.map((event, idx) => (
                  <div
                    key={idx}
                    className={`${
                      event.type.includes('error') || event.type.includes('fail') ? 'text-red-400' :
                      event.type.includes('warn') ? 'text-yellow-400' :
                      event.type.includes('start') || event.type.includes('creat') ? 'text-blue-400' :
                      event.type.includes('complet') || event.type.includes('success') ? 'text-green-400' :
                      'text-gray-300'
                    }`}
                  >
                    [{new Date(event.created_at).toLocaleString()}] [{event.type}]{event.worker_id ? ` [${event.worker_id}]` : ''} {event.summary}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Code Viewer Modal */}
      {selectedCode && (
        <CodeViewer
          experimentId={selectedCode}
          apiUrl={API_URL}
          onClose={() => setSelectedCode(null)}
        />
      )}

      {/* Artifacts Modal */}
      {selectedArtifacts && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedArtifacts(null)}>
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">
                Artifacts: {selectedArtifacts.id}
              </h3>
              <button
                onClick={() => setSelectedArtifacts(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl leading-none"
              >
                ×
              </button>
            </div>
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
              <div className="space-y-6">
                {/* Proposal */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Proposal</h4>
                  <button
                    onClick={() => navigate(`/proposals/${selectedArtifacts.proposal_id}`)}
                    className="text-blue-600 hover:text-blue-800 underline text-sm"
                  >
                    View {selectedArtifacts.proposal_id}
                  </button>
                </div>

                {/* Results */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Results</h4>
                  {selectedArtifacts.results ? (
                    <div className="bg-gray-50 p-3 rounded text-xs">
                      <pre className="overflow-auto max-h-40">
                        {JSON.stringify(selectedArtifacts.results, null, 2)}
                      </pre>
                      {selectedArtifacts.wandb_url && (
                        <div className="mt-2">
                          <a href={selectedArtifacts.wandb_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 underline">
                            View on W&B →
                          </a>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No results available</p>
                  )}
                </div>

                {/* Code */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Code</h4>
                  {selectedArtifacts.code_files.length > 0 ? (
                    <button
                      onClick={() => {
                        setSelectedArtifacts(null)
                        setSelectedCode(selectedArtifacts.id)
                      }}
                      className="text-blue-600 hover:text-blue-800 underline text-sm"
                    >
                      Browse Code ({selectedArtifacts.code_files.length} files)
                    </button>
                  ) : (
                    <p className="text-sm text-gray-500">No code available</p>
                  )}
                </div>

                {/* Logs */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Logs</h4>
                  <button
                    onClick={() => {
                      setSelectedArtifacts(null)
                      fetchLog(selectedArtifacts.proposal_id)
                    }}
                    className="text-blue-600 hover:text-blue-800 underline text-sm"
                  >
                    View Logs
                  </button>
                </div>

                {/* Artifacts Download */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Artifacts (S3)</h4>
                  {selectedArtifacts.artifacts_url ? (
                    <a href={selectedArtifacts.artifacts_url} download className="text-blue-600 hover:text-blue-800 underline text-sm">
                      Download artifacts.tar.gz
                    </a>
                  ) : (
                    <p className="text-sm text-gray-500">Not yet available (waiting for experiment to complete)</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
