import { useState, useEffect, useRef } from 'react'

interface LogViewerProps {
  apiUrl: string
}

export default function LogViewer({ apiUrl }: LogViewerProps) {
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [autoScroll, setAutoScroll] = useState(true)
  const logContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Fetch initial logs using events endpoint
    const fetchLogs = async () => {
      try {
        const res = await fetch(`${apiUrl}/events?limit=100`)
        if (res.ok) {
          const events = await res.json()
          // Format events as log lines
          const logLines = events.map((event: { created_at: string; type: string; summary: string; experiment_id?: string }) =>
            `[${new Date(event.created_at).toLocaleString()}] [${event.type}]${event.experiment_id ? ` [${event.experiment_id}]` : ''} ${event.summary}`
          )
          setLogs(logLines || [])
        }
      } catch (err) {
        console.error('Error fetching logs:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchLogs()

    // Subscribe to SSE stream for realtime updates
    const eventSource = new EventSource(`${apiUrl}/events/stream`)
    eventSource.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data)
        const line = `[${new Date(event.created_at).toLocaleString()}] [${event.type}]${event.experiment_id ? ` [${event.experiment_id}]` : ''} ${event.summary}`
        setLogs(prev => [...prev, line])
      } catch {
        // ignore unparseable messages
      }
    }

    return () => eventSource.close()
  }, [apiUrl])

  // Auto-scroll to bottom when new logs arrive (since newest is at bottom)
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleScroll = () => {
    if (!logContainerRef.current) return

    const { scrollTop, scrollHeight, clientHeight } = logContainerRef.current
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50

    setAutoScroll(isAtBottom)
  }

  if (loading) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">
        Loading logs...
      </div>
    )
  }

  if (logs.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">
        No logs available
      </div>
    )
  }

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <div className="bg-gray-800 px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-gray-300">runner.log</span>
          <span className="text-xs bg-gray-700 text-gray-400 px-2 py-0.5 rounded">
            Oldest → Newest
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-400">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="rounded"
            />
            Auto-scroll to bottom
          </label>
          <span>{logs.length} lines</span>
        </div>
      </div>

      <div
        ref={logContainerRef}
        onScroll={handleScroll}
        className="bg-gray-900 p-4 overflow-auto max-h-96 font-mono text-xs"
      >
        {logs.map((line, idx) => (
          <div
            key={idx}
            className={`${
              line.includes('ERROR') ? 'text-red-400' :
              line.includes('WARNING') ? 'text-yellow-400' :
              line.includes('INFO') ? 'text-blue-400' :
              'text-gray-300'
            }`}
          >
            {line}
          </div>
        ))}
      </div>

      <div className="bg-gray-800 px-4 py-2 flex items-center justify-between text-xs text-gray-400">
        <span>Showing last {logs.length} events (newest at bottom)</span>
        <button
          onClick={() => window.open(`${apiUrl}/events?limit=1000`, '_blank')}
          className="text-blue-400 hover:text-blue-300 underline"
        >
          View full log
        </button>
      </div>
    </div>
  )
}
