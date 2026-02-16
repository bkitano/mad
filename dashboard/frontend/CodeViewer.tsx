import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface FileNode {
  name: string
  path: string
  type: 'file' | 'directory'
  size?: number
  children?: FileNode[]
}

interface CodeViewerProps {
  experimentId: string
  sseUrl: string
  onClose: () => void
  embedded?: boolean
}

export default function CodeViewer({ experimentId, sseUrl, onClose, embedded = false }: CodeViewerProps) {
  const [fileTree, setFileTree] = useState<FileNode | null>(null)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [fileContent, setFileContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set())

  // Extract experiment number from proposal ID (e.g., "029-some-name" -> "029")
  const experimentNumber = experimentId.split('-')[0]

  useEffect(() => {
    fetchFileTree()
    // Auto-expand root level
    setExpandedPaths(new Set(['']))
  }, [experimentId])

  const fetchFileTree = async () => {
    try {
      setLoading(true)
      const res = await fetch(`${sseUrl}/api/experiment-code/${experimentNumber}`)
      if (res.ok) {
        const data = await res.json()
        setFileTree(data.file_tree)
      } else if (res.status === 404) {
        setError('Code not available yet - experiment may still be setting up')
      } else {
        setError('Failed to load code structure')
      }
    } catch (err) {
      setError(`Error: ${err}`)
    } finally {
      setLoading(false)
    }
  }

  const fetchFile = async (path: string) => {
    try {
      setLoading(true)
      setSelectedFile(path)
      const res = await fetch(`${sseUrl}/api/experiment-code/${experimentNumber}/file?path=${encodeURIComponent(path)}`)
      if (res.ok) {
        const data = await res.json()
        setFileContent(data.content)
      } else {
        setFileContent(`# Error\n\nFailed to load file: ${path}`)
      }
    } catch (err) {
      setFileContent(`# Error\n\nError loading file: ${err}`)
    } finally {
      setLoading(false)
    }
  }

  const toggleExpanded = (path: string) => {
    setExpandedPaths(prev => {
      const next = new Set(prev)
      if (next.has(path)) {
        next.delete(path)
      } else {
        next.add(path)
      }
      return next
    })
  }

  const renderFileTree = (node: FileNode, depth: number = 0): JSX.Element => {
    const isDirectory = node.type === 'directory'
    const isExpanded = expandedPaths.has(node.path)

    return (
      <div key={node.path}>
        <div
          className={`flex items-center gap-1 px-2 py-1 hover:bg-gray-100 cursor-pointer ${
            selectedFile === node.path ? 'bg-blue-50 text-blue-700' : ''
          }`}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
          onClick={() => {
            if (isDirectory) {
              toggleExpanded(node.path)
            } else {
              fetchFile(node.path)
            }
          }}
        >
          <span className="text-xs">
            {isDirectory ? (isExpanded ? '📂' : '📁') : '📄'}
          </span>
          <span className="text-sm font-mono">{node.name}</span>
          {!isDirectory && node.size && (
            <span className="text-xs text-gray-400 ml-auto">
              {formatBytes(node.size)}
            </span>
          )}
        </div>
        {isDirectory && isExpanded && node.children && (
          <div>
            {node.children.map((child) => renderFileTree(child, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
  }

  const getLanguage = (path: string) => {
    const ext = path.split('.').pop()?.toLowerCase()
    const langMap: Record<string, string> = {
      py: 'python',
      js: 'javascript',
      ts: 'typescript',
      jsx: 'jsx',
      tsx: 'tsx',
      yaml: 'yaml',
      yml: 'yaml',
      json: 'json',
      md: 'markdown',
      sh: 'bash',
      toml: 'toml',
      txt: 'text',
    }
    return langMap[ext || ''] || 'text'
  }

  const content = (
    <>
      {/* Header */}
      {!embedded && (
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            Code: {experimentId} ({experimentNumber})
          </h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-2xl leading-none"
          >
            ×
          </button>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* File Tree Sidebar */}
        <div className="w-64 border-r border-gray-200 overflow-y-auto bg-gray-50">
          {loading && !fileTree && (
            <div className="p-4 text-sm text-gray-500">Loading...</div>
          )}
          {error && (
            <div className="p-4">
              <div className="text-sm text-amber-600 mb-2">⚠️ Code Not Available</div>
              <div className="text-xs text-gray-600">{error}</div>
              <div className="text-xs text-gray-500 mt-3">
                The experiment agent may still be:
                <ul className="list-disc ml-4 mt-1">
                  <li>Creating the code directory</li>
                  <li>Implementing the architecture</li>
                  <li>Setting up dependencies</li>
                </ul>
                <div className="mt-2">Check back in a few minutes!</div>
              </div>
            </div>
          )}
          {fileTree && renderFileTree(fileTree)}
        </div>

        {/* File Content */}
        <div className="flex-1 overflow-y-auto">
          {!selectedFile && (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <p className="text-lg mb-2">👈 Select a file to view</p>
                <p className="text-sm">Click on any file in the tree to see its contents</p>
              </div>
            </div>
          )}
          {selectedFile && fileContent && (
            <div className="p-4">
              <div className="mb-2 text-xs text-gray-500 font-mono">
                {selectedFile}
              </div>
              {getLanguage(selectedFile) === 'markdown' ? (
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown>{fileContent}</ReactMarkdown>
                </div>
              ) : (
                <SyntaxHighlighter
                  language={getLanguage(selectedFile)}
                  style={vscDarkPlus}
                  showLineNumbers={true}
                  customStyle={{
                    margin: 0,
                    borderRadius: '0.5rem',
                    fontSize: '0.875rem',
                  }}
                  lineNumberStyle={{
                    minWidth: '3em',
                    paddingRight: '1em',
                    color: '#6e7681',
                    userSelect: 'none',
                  }}
                >
                  {fileContent}
                </SyntaxHighlighter>
              )}
            </div>
          )}
          {loading && selectedFile && (
            <div className="flex items-center justify-center h-full">
              <div className="text-gray-500">Loading file...</div>
            </div>
          )}
        </div>
      </div>
    </>
  )

  if (embedded) {
    return (
      <div className="bg-white h-full flex flex-col">
        {content}
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={onClose}>
      <div className="bg-white rounded-lg shadow-xl w-full max-w-7xl h-[90vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
        {content}
      </div>
    </div>
  )
}
