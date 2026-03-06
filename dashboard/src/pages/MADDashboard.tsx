import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import ExperimentCard from '../components/mad/ExperimentCard'
import AgentStatus from '../components/mad/AgentStatus'
import LogViewer from '../components/mad/LogViewer'
import ProposalsView from '../components/mad/ProposalsView'
import TricksView from '../components/mad/TricksView'
import ResearchLog from '../components/mad/ResearchLog'
import CodeViewer from '../components/mad/CodeViewer'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface ActiveExperiment {
  agent_id: string
  proposal_id: string
  started_at: string
  last_heartbeat: string
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

type TabType = 'experiments' | 'proposals' | 'tricks' | 'log'

const API_URL = 'https://mad.briankitano.com'

export default function MADDashboard() {
  const navigate = useNavigate()
  const location = useLocation()

  // Determine active tab from path
  const getActiveTab = (): TabType => {
    const path = location.pathname
    if (path.includes('/proposals')) return 'proposals'
    if (path.includes('/tricks')) return 'tricks'
    if (path.includes('/log')) return 'log'
    return 'experiments'
  }
  const activeTab = getActiveTab()

  const [data, setData] = useState<DashboardData | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [selectedResult, setSelectedResult] = useState<{ id: string; content: string } | null>(null)
  const [selectedLog, setSelectedLog] = useState<{ id: string; content: string } | null>(null)
  const [selectedCode, setSelectedCode] = useState<string | null>(null)
  const [isInitialLoad, setIsInitialLoad] = useState(true)
  const [showDescription, setShowDescription] = useState(true)

  // Fetch initial state immediately
  useEffect(() => {
    const fetchInitialState = async () => {
      try {
        console.log('Fetching initial state...')
        // Fetch both claims (active work) and stats
        const [claimsRes, statsRes] = await Promise.all([
          fetch(`${API_URL}/claims?status=active`),
          fetch(`${API_URL}/stats`)
        ])

        if (claimsRes.ok && statsRes.ok) {
          const claims = await claimsRes.json()
          // Stats response available for future use
          await statsRes.json()

          // Transform claims into active_work format
          const activeWork: Record<string, ActiveExperiment> = {}
          for (const claim of claims) {
            activeWork[claim.proposal_id] = {
              agent_id: claim.agent_id,
              proposal_id: claim.proposal_id,
              started_at: claim.claimed_at,
              last_heartbeat: claim.last_heartbeat,
              status: claim.status
            }
          }

          setData({ active_work: activeWork, history: [] })
          setLastUpdate(new Date())
          setIsInitialLoad(false)
          console.log('Initial state loaded')
        }
      } catch (err) {
        console.error('Error fetching initial state:', err)
        // Don't set error - SSE will handle it
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
          console.log('Received event:', event.event_type)

          // Handle claim-related events to update active work
          if (event.event_type === 'claim.acquired' || event.event_type === 'claim.released') {
            // Refetch claims on claim changes
            fetch(`${API_URL}/claims?status=active`)
              .then(res => res.json())
              .then(claims => {
                const activeWork: Record<string, ActiveExperiment> = {}
                for (const claim of claims) {
                  activeWork[claim.proposal_id] = {
                    agent_id: claim.agent_id,
                    proposal_id: claim.proposal_id,
                    started_at: claim.claimed_at,
                    last_heartbeat: claim.last_heartbeat,
                    status: claim.status
                  }
                }
                setData(prev => ({ ...prev, active_work: activeWork } as DashboardData))
                setLastUpdate(new Date())
              })
              .catch(err => console.error('Error refetching claims:', err))
          }

          // Handle experiment events
          if (event.event_type?.startsWith('experiment.')) {
            setLastUpdate(new Date())
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
      const res = await fetch(`${API_URL}/experiments/${experimentId}/events?limit=100`)
      if (res.ok) {
        const events = await res.json()
        // Format events as markdown log
        const logContent = events.map((event: { timestamp: string; event_type: string; summary: string; agent?: string }) =>
          `**${new Date(event.timestamp).toLocaleString()}** - [${event.event_type}] ${event.summary}${event.agent ? ` (${event.agent})` : ''}`
        ).join('\n\n')
        setSelectedLog({ id: proposalId, content: logContent || '# No Events\n\nNo events recorded for this experiment yet.' })
      } else {
        const message = `# Log Not Available\n\nNo experiment log found for ${experimentId} (${proposalId}).\n\nThe experiment may not have started yet or logs haven't been created.`
        setSelectedLog({ id: proposalId, content: message })
      }
    } catch (err) {
      console.error('Error fetching log:', err)
      setSelectedLog({ id: proposalId, content: `# Error\n\nFailed to load log: ${err}` })
    }
  }

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

  // Calculate stats
  const activeExperiments = data ? Object.keys(data.active_work).length : 0
  const recentHistory = data?.history?.slice(0, 5) || []
  const completedToday = data?.history?.filter(item => {
    if (!item.completed_at) return false
    const completed = new Date(item.completed_at)
    const today = new Date()
    return completed.toDateString() === today.toDateString()
  }).length || 0

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">😡 MAD Architecture Search</h1>
        <p className="mt-2 text-gray-600">
          Real-time experiment monitoring dashboard
        </p>
      </div>

      {/* About Section - show on experiments tab */}
      {activeTab === 'experiments' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg overflow-hidden">
          <button
            onClick={() => setShowDescription(!showDescription)}
            className="w-full px-6 py-4 flex items-center justify-between hover:bg-blue-100 transition-colors"
          >
            <h3 className="text-lg font-semibold text-gray-900">About This Dashboard</h3>
            <svg
              className={`w-5 h-5 text-gray-600 transition-transform ${showDescription ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showDescription && (
            <div className="px-6 pb-6 text-sm text-gray-700 space-y-3">
              <p>
                <strong>MAD (Multi-Agent Design)</strong> is an experiment in autonomous research. The goal is to see if AI agents can
                explore novel algorithms for efficient sequence modeling at the intersection of three dimensions:
              </p>

              <ul className="list-disc list-inside space-y-1 ml-4 text-gray-700">
                <li><strong>Hardware Efficiency:</strong> Memory management patterns and optimized kernel implementations</li>
                <li><strong>Computational Complexity:</strong> Algorithmic techniques and approximations to reduce computation</li>
                <li><strong>Theoretical Expressivity:</strong> State tracking, associative recall, and representational capacity</li>
              </ul>

              <div>
                <p className="font-medium text-gray-900 mb-1">The approach:</p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li><strong>Research Agent:</strong> Tries to extract core ideas from papers as reusable "tricks," then combine them in novel ways to generate proposals with minimal experiments to show signs of life</li>
                  <li><strong>Experiment Agents:</strong> Attempt to autonomously implement proposals, write training code, and run experiments on Modal's serverless GPUs</li>
                  <li><strong>Evaluation:</strong> Run small-scale language modeling benchmarks to see if ideas show promise worth exploring further</li>
                </ul>
              </div>

              <p>
                This dashboard streams live updates via Server-Sent Events (SSE), letting you watch the process unfold
                in real-time—from paper analysis to proposal generation to GPU experiments—with minimal human intervention.
              </p>

              <div className="pt-2 border-t border-blue-200 space-y-2">
                <p className="text-xs text-gray-600">
                  <strong>Tech stack:</strong> Built with <a href="https://modal.com" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Modal</a> (serverless GPU compute)
                  and <a href="https://github.com/anthropics/anthropic-sdk-python" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Claude Agents SDK</a> (autonomous agent orchestration).
                  Running continuously on Modal's infrastructure.
                </p>
                <p className="text-xs text-gray-600">
                  <strong>🔗 Public Repository:</strong> <a href="https://github.com/bkitano/mad" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">github.com/bkitano/mad</a>
                </p>
                <p className="text-xs text-gray-600 italic">
                  An exploration of how far we can push AI agents in scientific discovery: can they propose ideas, write code,
                  debug failures, and iterate on results autonomously? We're finding out.
                </p>
                <p className="text-xs text-purple-700 font-medium">
                  💜 This project burns through ~800M tokens/day of Claude + GPU credits—if you'd like to sponsor compute time, I'd love Modal/Anthropic credits uwu
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex gap-8">
          <button
            onClick={() => navigate('/mad')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'experiments'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Experiments
          </button>
          <button
            onClick={() => navigate('/mad/proposals')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'proposals'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Proposals
          </button>
          <button
            onClick={() => navigate('/mad/tricks')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'tricks'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Tricks
          </button>
          <button
            onClick={() => navigate('/mad/log')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'log'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Research Log
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
                  <div className="text-2xl font-bold text-gray-900">{recentHistory.length}</div>
                  <div className="text-sm text-gray-700">Recent Experiments</div>
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
                      navigate(`/mad/proposals/${proposalId}`)
                    }}
                    onViewLog={fetchLog}
                    onViewCode={setSelectedCode}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Agent Status */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Agent Health</h2>
            {isInitialLoad ? (
              <div className="bg-gray-50 p-6 rounded-lg animate-pulse space-y-4">
                <div className="flex items-center gap-4">
                  <div className="h-4 w-4 bg-gray-300 rounded-full"></div>
                  <div className="h-4 bg-gray-300 rounded w-32"></div>
                  <div className="h-4 bg-gray-300 rounded w-24 ml-auto"></div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="h-4 w-4 bg-gray-300 rounded-full"></div>
                  <div className="h-4 bg-gray-300 rounded w-32"></div>
                  <div className="h-4 bg-gray-300 rounded w-24 ml-auto"></div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="h-4 w-4 bg-gray-300 rounded-full"></div>
                  <div className="h-4 bg-gray-300 rounded w-32"></div>
                  <div className="h-4 bg-gray-300 rounded w-24 ml-auto"></div>
                </div>
              </div>
            ) : (
              <AgentStatus apiUrl={API_URL} />
            )}
          </div>

          {/* Recent Logs */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Recent Logs</h2>
            <LogViewer apiUrl={API_URL} />
          </div>

          {/* Recent Experiments */}
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">Recent Experiments</h2>
            {isInitialLoad ? (
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
            ) : (
              <div className="space-y-2">
                {recentHistory.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => fetchResult(item.proposal_id)}
                    className="w-full flex items-center justify-between p-3 bg-gray-50 hover:bg-gray-100 rounded-lg text-sm transition-colors cursor-pointer"
                  >
                    <div>
                      <span className="font-mono text-xs bg-gray-200 px-2 py-0.5 rounded mr-2">
                        {item.agent_id}
                      </span>
                      <span className="text-gray-900">{item.proposal_id}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        item.status === 'completed' ? 'bg-green-100 text-green-800' :
                        item.status === 'failed' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {item.status}
                      </span>
                      {item.completed_at && (
                        <span className="text-gray-500 text-xs">
                          {new Date(item.completed_at).toLocaleString()}
                        </span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
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
      {activeTab === 'log' && <ResearchLog apiUrl={API_URL} />}

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
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">
                Experiment Log: {selectedLog.id}
              </h3>
              <button
                onClick={() => setSelectedLog(null)}
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
                  {selectedLog.content}
                </ReactMarkdown>
              </div>
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

    </div>
  )
}
