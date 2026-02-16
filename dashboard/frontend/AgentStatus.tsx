import { useState, useEffect } from 'react'
import { Activity, Clock, CheckCircle, AlertCircle } from 'lucide-react'

interface AgentInfo {
  status: 'running' | 'waiting' | 'idle' | 'error'
  last_update: string
  details: {
    iteration?: number
    interval_minutes?: number
    next_run?: number
    num_parallel_agents?: number
  }
}

interface AgentStatusProps {
  sseUrl: string
}

export default function AgentStatus({ sseUrl }: AgentStatusProps) {
  const [agents, setAgents] = useState<Record<string, AgentInfo>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${sseUrl}/api/agent-status`)
        if (res.ok) {
          const data = await res.json()
          setAgents(data.agents || {})
        }
      } catch (err) {
        console.error('Error fetching agent status:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchStatus()

    // Refresh every 5 seconds
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [sseUrl])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'waiting':
        return 'bg-blue-100 text-blue-800 border-blue-300'
      case 'idle':
        return 'bg-gray-100 text-gray-800 border-gray-300'
      case 'error':
        return 'bg-red-100 text-red-800 border-red-300'
      default:
        return 'bg-gray-100 text-gray-600 border-gray-300'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <Activity className="w-4 h-4" />
      case 'waiting':
        return <Clock className="w-4 h-4" />
      case 'idle':
        return <CheckCircle className="w-4 h-4" />
      case 'error':
        return <AlertCircle className="w-4 h-4" />
      default:
        return null
    }
  }

  const formatTimeUntil = (timestamp: number) => {
    const now = Date.now() / 1000
    const seconds = Math.max(0, timestamp - now)
    const minutes = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)

    if (minutes > 0) {
      return `${minutes}m ${secs}s`
    }
    return `${secs}s`
  }

  const getAgentLabel = (agentType: string) => {
    const labels: Record<string, string> = {
      trick_search: 'Trick Search',
      research: 'Research',
      experiment: 'Experiments',
      log: 'Log Generator',
      scaler: 'Experiment Scaler'
    }
    return labels[agentType] || agentType
  }

  if (loading) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg text-center text-gray-500 text-sm">
        Loading agent status...
      </div>
    )
  }

  const agentTypes = ['trick_search', 'research', 'experiment', 'log', 'scaler']
  const activeAgents = agentTypes.filter(type => agents[type])

  if (activeAgents.length === 0) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg text-center text-gray-500 text-sm">
        No agent status available
      </div>
    )
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
        <Activity className="w-4 h-4" />
        Agent Status
      </h3>

      <div className="space-y-2">
        {agentTypes.map((agentType) => {
          const agent = agents[agentType]
          if (!agent) return null

          return (
            <div
              key={agentType}
              className={`p-3 rounded-lg border ${getStatusColor(agent.status)}`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getStatusIcon(agent.status)}
                  <span className="font-medium text-sm">
                    {getAgentLabel(agentType)}
                  </span>
                </div>
                <div className="text-xs font-semibold uppercase">
                  {agent.status}
                </div>
              </div>

              {agent.details && (
                <div className="mt-2 text-xs space-y-1">
                  {agent.details.iteration !== undefined && (
                    <div>Iteration: {agent.details.iteration}</div>
                  )}
                  {agent.details.num_parallel_agents !== undefined && (
                    <div>Agents: {agent.details.num_parallel_agents} parallel</div>
                  )}
                  {agent.details.next_run && agent.status === 'waiting' && (
                    <div>Next run: {formatTimeUntil(agent.details.next_run)}</div>
                  )}
                  {agent.details.interval_minutes && (
                    <div>Interval: {agent.details.interval_minutes} min</div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
