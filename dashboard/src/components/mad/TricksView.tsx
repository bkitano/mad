import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface Trick {
  id: number
  slug: string
  title: string
  category: string
  gain_type: string
  source?: string
  paper?: string
  documented?: string
  created_at?: string
}

interface TrickFull extends Trick {
  content: string
  updated_at?: string
}

interface TricksViewProps {
  apiUrl: string
}

const CATEGORY_COLORS: Record<string, string> = {
  decomposition: 'bg-blue-100 text-blue-800',
  parallelization: 'bg-purple-100 text-purple-800',
  kernel: 'bg-orange-100 text-orange-800',
  stability: 'bg-green-100 text-green-800',
  efficiency: 'bg-teal-100 text-teal-800',
  algebraic: 'bg-rose-100 text-rose-800',
  approximation: 'bg-amber-100 text-amber-800',
}

const GAIN_TYPE_COLORS: Record<string, string> = {
  efficiency: 'bg-emerald-100 text-emerald-800',
  expressivity: 'bg-violet-100 text-violet-800',
  accuracy: 'bg-sky-100 text-sky-800',
}

function Badge({ label, colorMap }: { label: string; colorMap: Record<string, string> }) {
  const color = colorMap[label] || 'bg-gray-100 text-gray-800'
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${color}`}>
      {label}
    </span>
  )
}

// Tricks List Component
function TricksList({ apiUrl }: TricksViewProps) {
  const navigate = useNavigate()
  const [tricks, setTricks] = useState<Trick[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [categoryFilter, setCategoryFilter] = useState<string>('')
  const [gainTypeFilter, setGainTypeFilter] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchTricks = async () => {
      try {
        setLoading(true)
        const params = new URLSearchParams()
        params.set('limit', '1000')
        if (searchTerm) params.set('search', searchTerm)
        if (categoryFilter) params.set('category', categoryFilter)
        if (gainTypeFilter) params.set('gain_type', gainTypeFilter)

        const res = await fetch(`${apiUrl}/tricks?${params}`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setTricks(data)
        setError(null)
      } catch (err) {
        console.error('Error fetching tricks:', err)
        setError('Failed to load tricks')
      } finally {
        setLoading(false)
      }
    }

    const debounce = setTimeout(fetchTricks, searchTerm ? 300 : 0)
    return () => clearTimeout(debounce)
  }, [apiUrl, searchTerm, categoryFilter, gainTypeFilter])

  // Extract unique categories and gain types for filters
  const categories = [...new Set(tricks.map(t => t.category))].sort()
  const gainTypes = [...new Set(tricks.map(t => t.gain_type))].sort()

  if (loading && tricks.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">
        Loading tricks...
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* About Tricks */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">What are Tricks?</h3>
        <p className="text-sm text-gray-700">
          Tricks are atomic, reusable insights extracted from research papers — specific CUDA kernel patterns,
          numerical stability techniques, or architectural components that serve as building blocks for new proposals.
        </p>
      </div>

      {error && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800">
          {error}
        </div>
      )}

      {/* Search & Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          placeholder="Search tricks..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-1 min-w-[200px] border border-gray-300 rounded px-3 py-2 text-sm"
        />
        <select
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
          className="border border-gray-300 rounded px-3 py-2 text-sm"
        >
          <option value="">All categories</option>
          {categories.map(c => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select
          value={gainTypeFilter}
          onChange={(e) => setGainTypeFilter(e.target.value)}
          className="border border-gray-300 rounded px-3 py-2 text-sm"
        >
          <option value="">All gain types</option>
          {gainTypes.map(g => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
        <div className="text-gray-500 text-sm">
          {tricks.length} trick{tricks.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Tricks Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {tricks.map(trick => (
          <button
            key={trick.id}
            onClick={() => navigate(`/tricks/${trick.slug}`)}
            className="text-left p-3 border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-xs text-gray-400">#{trick.id}</span>
              <Badge label={trick.category} colorMap={CATEGORY_COLORS} />
              <Badge label={trick.gain_type} colorMap={GAIN_TYPE_COLORS} />
            </div>
            <div className="text-sm font-medium text-gray-900 mb-1">
              {trick.title}
            </div>
            {trick.source && (
              <div className="text-xs text-gray-500 truncate">
                {trick.source}
              </div>
            )}
          </button>
        ))}
      </div>

      {tricks.length === 0 && !loading && (
        <div className="bg-gray-50 p-8 rounded-lg text-center text-gray-500">
          {searchTerm || categoryFilter || gainTypeFilter
            ? 'No tricks match your filters'
            : 'No tricks yet'}
        </div>
      )}
    </div>
  )
}

interface TrickDetailProps {
  apiUrl: string
  trickId: string
}

// Trick Detail Component
function TrickDetail({ apiUrl, trickId }: TrickDetailProps) {
  const navigate = useNavigate()
  const [trick, setTrick] = useState<TrickFull | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!trickId) return

    const fetchContent = async () => {
      try {
        const res = await fetch(`${apiUrl}/tricks/${encodeURIComponent(trickId)}`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        setTrick(data)
        setError(null)
      } catch (err) {
        console.error('Error fetching trick:', err)
        setError('Failed to load trick')
      } finally {
        setLoading(false)
      }
    }

    fetchContent()
  }, [trickId, apiUrl])

  if (loading) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">
        Loading trick...
      </div>
    )
  }

  return (
    <div>
      <button
        onClick={() => navigate('/tricks')}
        className="mb-4 text-blue-600 hover:text-blue-800 text-sm"
      >
        &larr; Back to tricks list
      </button>

      {error ? (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800">
          {error}
        </div>
      ) : trick ? (
        <div>
          {/* Header */}
          <div className="mb-4 pb-4 border-b border-gray-200">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-sm text-gray-400">#{trick.id}</span>
              <Badge label={trick.category} colorMap={CATEGORY_COLORS} />
              <Badge label={trick.gain_type} colorMap={GAIN_TYPE_COLORS} />
            </div>
            <h1 className="text-xl font-semibold text-gray-900 mb-2">{trick.title}</h1>
            {trick.source && (
              <p className="text-sm text-gray-600">{trick.source}</p>
            )}
            {trick.documented && (
              <p className="text-xs text-gray-400 mt-1">Documented: {trick.documented}</p>
            )}
          </div>

          {/* Content */}
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {trick.content}
            </ReactMarkdown>
          </div>
        </div>
      ) : null}
    </div>
  )
}

// Main Router Component
export default function TricksView({ apiUrl }: TricksViewProps) {
  const location = useLocation()

  const pathParts = location.pathname.split('/').filter(Boolean)
  const trickId = pathParts.length > 2 && pathParts[1] === 'tricks' ? pathParts[2] : null

  if (trickId) {
    return <TrickDetail apiUrl={apiUrl} trickId={trickId} />
  }

  return <TricksList apiUrl={apiUrl} />
}
