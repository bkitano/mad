import { useState } from 'react'

interface ExperimentCardProps {
  proposalId: string
  experiment: {
    id?: string
    proposal_id: string
    started_at: string
    last_heartbeat?: string
    status: string
    worker_id?: string
  }
  apiUrl: string
  onViewProposal?: (proposalId: string) => void
  onViewLog?: (experimentId: string) => void
  onViewCode?: (experimentId: string) => void
}

export default function ExperimentCard({ proposalId, experiment, onViewProposal, onViewLog, onViewCode }: ExperimentCardProps) {
  const [expanded, setExpanded] = useState(false)

  const startedAt = new Date(experiment.started_at)
  const lastHeartbeat = experiment.last_heartbeat ? new Date(experiment.last_heartbeat) : null
  const now = new Date()

  // Calculate time since last heartbeat
  const timeSinceHeartbeat = lastHeartbeat ? Math.floor((now.getTime() - lastHeartbeat.getTime()) / 1000) : null
  const isHealthy = timeSinceHeartbeat === null || timeSinceHeartbeat < 600 // 10 minutes

  // Calculate runtime
  const runtimeSeconds = Math.floor((now.getTime() - startedAt.getTime()) / 1000)
  const runtimeMinutes = Math.floor(runtimeSeconds / 60)
  const runtimeHours = Math.floor(runtimeMinutes / 60)

  const formatRuntime = () => {
    if (runtimeHours > 0) {
      return `${runtimeHours}h ${runtimeMinutes % 60}m`
    }
    return `${runtimeMinutes}m`
  }

  const formatHeartbeat = () => {
    if (timeSinceHeartbeat === null) return 'n/a'
    if (timeSinceHeartbeat < 60) return `${timeSinceHeartbeat}s ago`
    if (timeSinceHeartbeat < 3600) return `${Math.floor(timeSinceHeartbeat / 60)}m ago`
    return `${Math.floor(timeSinceHeartbeat / 3600)}h ago`
  }

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-lg font-medium text-gray-900">
              {proposalId}
            </h3>
            <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-yellow-500'}`} />
          </div>

          <div className="space-y-1 text-sm">
            <div className="flex items-center gap-4 text-gray-600">
              <span>Runtime: {formatRuntime()}</span>
              <span>•</span>
              <span>Last heartbeat: {formatHeartbeat()}</span>
              {experiment.worker_id && (
                <>
                  <span>•</span>
                  <span className="font-mono text-xs">{experiment.worker_id}</span>
                </>
              )}
            </div>
          </div>
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          className="text-sm text-blue-600 hover:text-blue-800"
        >
          {expanded ? 'Hide' : 'Details'}
        </button>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-gray-200 space-y-2 text-sm">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-gray-500">Started:</span>
              <div className="font-mono text-xs text-gray-900">
                {startedAt.toLocaleString()}
              </div>
            </div>
            <div>
              <span className="text-gray-500">Last Heartbeat:</span>
              <div className="font-mono text-xs text-gray-900">
                {lastHeartbeat?.toLocaleString() ?? 'n/a'}
              </div>
            </div>
          </div>

          <div>
            <span className="text-gray-500">Status:</span>
            <span className={`ml-2 px-2 py-0.5 rounded text-xs font-medium ${
              experiment.status === 'in_progress' ? 'bg-blue-100 text-blue-800' :
              experiment.status === 'starting' ? 'bg-yellow-100 text-yellow-800' :
              'bg-gray-100 text-gray-800'
            }`}>
              {experiment.status}
            </span>
          </div>

          <div className="mt-4 flex gap-2">
            {onViewProposal && (
              <button
                onClick={() => onViewProposal(proposalId)}
                className="px-3 py-1.5 text-xs font-medium text-purple-600 hover:text-purple-800 bg-purple-50 hover:bg-purple-100 rounded transition-colors"
              >
                View Proposal
              </button>
            )}
            {onViewLog && (
              <button
                onClick={() => onViewLog(experiment.id || proposalId)}
                className="px-3 py-1.5 text-xs font-medium text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 rounded transition-colors"
              >
                View Log
              </button>
            )}
            {onViewCode && (
              <button
                onClick={() => onViewCode(experiment.id || proposalId)}
                className="px-3 py-1.5 text-xs font-medium text-green-600 hover:text-green-800 bg-green-50 hover:bg-green-100 rounded transition-colors"
              >
                View Code
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
