import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { ChevronDown, ChevronRight } from 'lucide-react'

interface ResearchLogProps {
  sseUrl: string
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

export default function ResearchLog({ sseUrl }: ResearchLogProps) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set())
  const [logContents, setLogContents] = useState<Map<string, string>>(new Map())

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await fetch(`${sseUrl}/api/research-logs`)
        if (res.ok) {
          const data = await res.json()
          setLogs(data.logs || [])
          // Auto-expand the first (newest) log
          if (data.logs && data.logs.length > 0) {
            const newest = data.logs[0].filename
            setExpandedLogs(new Set([newest]))
            fetchLogContent(newest)
          }
        }
      } catch (err) {
        console.error('Error fetching research logs:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchLogs()

    // Refresh every 30 seconds
    const interval = setInterval(fetchLogs, 30000)
    return () => clearInterval(interval)
  }, [sseUrl])

  const fetchLogContent = async (filename: string) => {
    if (logContents.has(filename)) return

    try {
      const res = await fetch(`${sseUrl}/api/research-log/${filename}`)
      if (res.ok) {
        const data = await res.json()
        setLogContents(new Map(logContents).set(filename, data.content))
      }
    } catch (err) {
      console.error(`Error fetching log ${filename}:`, err)
    }
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
        Loading research logs...
      </div>
    )
  }

  if (logs.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">
        No research logs available
      </div>
    )
  }

  return (
    <div>
      <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="text-sm">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="font-semibold text-gray-900">Research Activity Log</span>
              <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded">
                {logs.length} updates
              </span>
              <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded">
                Newest first ↓
              </span>
            </div>
            <p className="text-gray-700 mb-2">
              Timestamped entries from the <strong>Research Agent</strong> documenting its analysis of experimental
              results, strategic decisions, and insights discovered during the architecture search process.
            </p>
            <p className="text-gray-700 text-xs">
              Each log entry shows what the Research Agent learned from recent experiments, which tricks it
              extracted, and what new proposals it generated. This creates a chronological record of the
              autonomous research process—watching the agent learn, adapt, and evolve its search strategy over time.
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
