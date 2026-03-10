import { useState, useEffect } from 'react'
import { Activity } from 'lucide-react'

interface AgentStatusProps {
  apiUrl: string
}

export default function AgentStatus({ apiUrl }: AgentStatusProps) {
  const [runningCount, setRunningCount] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${apiUrl}/experiments?status=running`)
        if (res.ok) {
          const exps = await res.json()
          setRunningCount(exps.length)
        }
      } catch (err) {
        console.error('Error fetching agent status:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [apiUrl])

  if (loading) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg text-center text-gray-500 text-sm">
        Loading status...
      </div>
    )
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
        <Activity className="w-4 h-4" />
        Status
      </h3>
      <div className={`p-3 rounded-lg border ${runningCount > 0 ? 'bg-green-100 text-green-800 border-green-300' : 'bg-gray-100 text-gray-800 border-gray-300'}`}>
        <div className="flex items-center justify-between">
          <span className="font-medium text-sm">Running Experiments</span>
          <span className="text-lg font-bold">{runningCount}</span>
        </div>
      </div>
    </div>
  )
}
