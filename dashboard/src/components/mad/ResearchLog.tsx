import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { ChevronDown, ChevronRight } from 'lucide-react'

interface ResearchLogProps {
  apiUrl: string
}

interface LogEntry {
  filename: string
  modified: number
  title?: string
  date?: string
  timestamp?: string
  tricks?: string
  proposals?: string
  experiments?: string
}

export default function ResearchLog({ apiUrl }: ResearchLogProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set())
  const [logContents, setLogContents] = useState<Map<string, string>>(new Map())
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        // Research logs endpoint not implemented in new API
        // Use events as a fallback to show recent activity
        const res = await fetch(`${apiUrl}/events?limit=50`)
        if (res.ok) {
          const events = await res.json()
          // Group events by date to simulate log entries
          const groupedByDate: Record<string, typeof events> = {}
          for (const event of events) {
            const date = new Date(event.created_at).toDateString()
            if (!groupedByDate[date]) {
              groupedByDate[date] = []
            }
            groupedByDate[date].push(event)
          }

          // Convert to log entries
          const logEntries: LogEntry[] = Object.entries(groupedByDate).map(([date, dateEvents]) => ({
            filename: date,
            modified: new Date(date).getTime(),
            title: `Activity on ${date}`,
            date,
            experiments: String(dateEvents.filter((e: { type: string }) => e.type.startsWith('experiment.')).length)
          }))

          setLogs(logEntries)

          // Store content for each log entry
          const newContents = new Map(logContents)
          for (const [date, dateEvents] of Object.entries(groupedByDate)) {
            const content = (dateEvents as Array<{ created_at: string; type: string; summary: string; experiment_id?: string }>)
              .map(e => `**${new Date(e.created_at).toLocaleTimeString()}** - [${e.type}] ${e.summary}`)
              .join('\n\n')
            newContents.set(date, content)
          }
          setLogContents(newContents)

          // Auto-expand first entry
          if (logEntries.length > 0) {
            setExpandedLogs(new Set([logEntries[0].filename]))
          }
        } else {
          setError('Failed to load activity logs')
        }
      } catch (err) {
        console.error('Error fetching research logs:', err)
        setError('Failed to load activity logs')
      } finally {
        setLoading(false)
      }
    }

    fetchLogs()

    // Refresh every 30 seconds
    const interval = setInterval(fetchLogs, 30000)
    return () => clearInterval(interval)
  }, [apiUrl])

  const fetchLogContent = async (_filename: string) => {
    // Content is already loaded in the main fetch
    return
  }

  const toggleLog = (filename: string) => {
    const newExpanded = new Set(expandedLogs)
    if (newExpanded.has(filename)) {
      newExpanded.delete(filename)
    } else {
      newExpanded.add(filename)
      fetchLogContent(filename)
    }
    setExpandedLogs(newExpanded)
  }

  const formatTimestamp = (filename: string) => {
    // Parse ISO-like timestamp from filename: 2026-02-15T03-52-00.md
    const match = filename.match(/(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})/)
    if (match) {
      const [, year, month, day, hour, minute, second] = match
      const date = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`)
      return date.toLocaleString()
    }
    return filename
  }

  if (loading) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">
        Loading activity logs...
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800">
        {error}
      </div>
    )
  }

  if (logs.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">
        No activity logs available
      </div>
    )
  }

  return (
    <div>
      <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-sm">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="font-semibold text-gray-900">Activity Log</span>
              <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded">
                {logs.length} days
              </span>
              <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded">
                Newest first ↓
              </span>
            </div>
            <p className="text-gray-700 mb-2">
              Timestamped events from the experiment execution system, showing experiment progress,
              status changes, and agent activity.
            </p>
            <p className="text-gray-700 text-xs">
              Each entry shows events grouped by day, including experiment creation, status updates,
              and completion events.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {logs.map((log) => {
          const isExpanded = expandedLogs.has(log.filename)
          const content = logContents.get(log.filename)

          return (
            <div
              key={log.filename}
              className="border border-gray-200 rounded-lg overflow-hidden"
            >
              <button
                onClick={() => toggleLog(log.filename)}
                className="w-full px-4 py-3 bg-gray-50 hover:bg-gray-100 flex items-center justify-between text-left transition-colors"
              >
                <div className="flex items-center gap-3">
                  {isExpanded ? (
                    <ChevronDown className="w-4 h-4 text-gray-600" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-gray-600" />
                  )}
                  <div>
                    <div className="font-medium text-gray-900">
                      {formatTimestamp(log.filename)}
                    </div>
                    {(log.tricks || log.proposals || log.experiments) && (
                      <div className="text-xs text-gray-600 mt-1 flex gap-3">
                        {log.tricks && <span>{log.tricks} tricks</span>}
                        {log.proposals && <span>{log.proposals} proposals</span>}
                        {log.experiments && <span>{log.experiments} experiments</span>}
                      </div>
                    )}
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  {isExpanded ? 'Click to collapse' : 'Click to expand'}
                </div>
              </button>

              {isExpanded && (
                <div className="p-4 bg-white border-t border-gray-200">
                  {content ? (
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <div className="text-gray-500 text-sm">Loading...</div>
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
