import { useState, useEffect } from 'react'
import { Routes, Route, useNavigate, useParams } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface Trick {
  id: string
  filename: string
  title?: string
  modified: number
}

interface TricksViewProps {
  sseUrl: string
}

// Tricks List Component
function TricksList({ sseUrl }: TricksViewProps) {
  const navigate = useNavigate()
  const [tricks, setTricks] = useState<Trick[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')

  // Fetch tricks list
  useEffect(() => {
    const fetchTricks = async () => {
      try {
        const res = await fetch(`${sseUrl}/api/tricks`)
        if (res.ok) {
          const data = await res.json()
          setTricks(data.tricks || [])
        }
      } catch (err) {
        console.error('Error fetching tricks:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchTricks()
  }, [sseUrl])

  // Filter tricks by search term
  const filteredTricks = tricks.filter(t => {
    if (!searchTerm) return true
    const search = searchTerm.toLowerCase()
    return (
      t.id.toLowerCase().includes(search) ||
      (t.title && t.title.toLowerCase().includes(search))
    )
  })

  if (loading) {
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
        <p className="text-sm text-gray-700 mb-2">
          Tricks are atomic, reusable insights extracted from research papers. Think of them as a curated
          knowledge base of techniques, optimizations, and implementation patterns from the broader ML literature.
        </p>
        <p className="text-sm text-gray-700">
          The Research Agent reads papers and distills key ideas into self-contained "tricks"—things like
          specific CUDA kernel patterns, numerical stability techniques, or architectural components.
          These tricks become building blocks that the agent can mix and combine when generating new proposals,
          allowing it to draw from the collective knowledge of the research community.
        </p>
      </div>

      {/* Search */}
      <div className="flex items-center gap-4">
        <input
          type="text"
          placeholder="Search tricks..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm"
        />
        <div className="text-gray-500 text-sm">
          {filteredTricks.length} trick{filteredTricks.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Tricks List */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {filteredTricks.map(trick => (
          <button
            key={trick.id}
            onClick={() => navigate(`/mad/tricks/${trick.id}`)}
            className="text-left p-3 border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
          >
            <div className="font-mono text-xs text-gray-500 mb-1">
              {trick.id}
            </div>
            <div className="text-sm text-gray-900">
              {trick.title || trick.id}
            </div>
          </button>
        ))}
      </div>

      {filteredTricks.length === 0 && (
        <div className="bg-gray-50 p-8 rounded-lg text-center text-gray-500">
          No tricks match "{searchTerm}"
        </div>
      )}
    </div>
  )
}

// Trick Detail Component
function TrickDetail({ sseUrl }: TricksViewProps) {
  const navigate = useNavigate()
  const { trickId } = useParams<{ trickId: string }>()
  const [trickContent, setTrickContent] = useState<string>('')
  const [loading, setLoading] = useState(true)

  // Fetch trick content
  useEffect(() => {
    if (!trickId) return

    const fetchContent = async () => {
      try {
        const res = await fetch(`${sseUrl}/api/trick/${trickId}`)
        if (res.ok) {
          const data = await res.json()
          setTrickContent(data.content || '')
        }
      } catch (err) {
        console.error('Error fetching trick content:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchContent()
  }, [trickId, sseUrl])

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
        onClick={() => navigate('/mad/tricks')}
        className="mb-4 text-blue-600 hover:text-blue-800 text-sm"
      >
        ← Back to tricks list
      </button>

      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex]}
        >
          {trickContent}
        </ReactMarkdown>
      </div>
    </div>
  )
}

// Main Router Component
export default function TricksView({ sseUrl }: TricksViewProps) {
  return (
    <Routes>
      <Route path="/" element={<TricksList sseUrl={sseUrl} />} />
      <Route path="/:trickId" element={<TrickDetail sseUrl={sseUrl} />} />
    </Routes>
  )
}
