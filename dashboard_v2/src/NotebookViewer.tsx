import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface NotebookCell {
  cell_type: 'code' | 'markdown' | 'raw'
  source: string[]
  outputs?: CellOutput[]
  execution_count?: number | null
}

interface CellOutput {
  output_type: string
  text?: string[]
  data?: Record<string, string | string[]>
  ename?: string
  evalue?: string
  traceback?: string[]
}

interface Notebook {
  cells: NotebookCell[]
  metadata?: {
    kernelspec?: { language?: string }
    language_info?: { name?: string }
  }
}

function joinText(text: string | string[]): string {
  return Array.isArray(text) ? text.join('') : text
}

function OutputDisplay({ output }: { output: CellOutput }) {
  if (output.output_type === 'stream' && output.text) {
    return (
      <pre className="nb-output-text">{joinText(output.text)}</pre>
    )
  }

  if (output.output_type === 'error' && output.traceback) {
    return (
      <pre className="nb-output-error">
        {output.traceback.join('\n').replace(/\x1b\[[0-9;]*m/g, '')}
      </pre>
    )
  }

  if ((output.output_type === 'display_data' || output.output_type === 'execute_result') && output.data) {
    const data = output.data
    if (data['image/png']) {
      return <img src={`data:image/png;base64,${joinText(data['image/png'])}`} className="nb-output-img" />
    }
    if (data['image/jpeg']) {
      return <img src={`data:image/jpeg;base64,${joinText(data['image/jpeg'])}`} className="nb-output-img" />
    }
    if (data['image/svg+xml']) {
      return <div className="nb-output-svg" dangerouslySetInnerHTML={{ __html: joinText(data['image/svg+xml']) }} />
    }
    if (data['text/html']) {
      return <div className="nb-output-html" dangerouslySetInnerHTML={{ __html: joinText(data['text/html']) }} />
    }
    if (data['text/plain']) {
      return <pre className="nb-output-text">{joinText(data['text/plain'])}</pre>
    }
  }

  return null
}

export default function NotebookViewer({ content }: { content: string }) {
  let notebook: Notebook
  try {
    notebook = JSON.parse(content)
  } catch {
    return <pre className="nb-error">Failed to parse notebook JSON</pre>
  }

  const language = notebook.metadata?.language_info?.name
    || notebook.metadata?.kernelspec?.language
    || 'python'

  return (
    <div className="nb-root">
      {notebook.cells.map((cell, i) => (
        <div key={i} className="nb-cell">
          {cell.cell_type === 'markdown' ? (
            <div className="nb-markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                {joinText(cell.source)}
              </ReactMarkdown>
            </div>
          ) : cell.cell_type === 'code' ? (
            <>
              <div className="nb-code-input">
                <div className="nb-execution-count">
                  [{cell.execution_count ?? ' '}]
                </div>
                <div className="nb-code-source">
                  <SyntaxHighlighter
                    language={language}
                    style={oneDark}
                    customStyle={{ margin: 0, padding: '12px', borderRadius: 0, background: '#0d1117', fontSize: '13px', lineHeight: '1.45' }}
                  >
                    {joinText(cell.source)}
                  </SyntaxHighlighter>
                </div>
              </div>
              {cell.outputs && cell.outputs.length > 0 && (
                <div className="nb-code-output">
                  {cell.outputs.map((output, j) => (
                    <OutputDisplay key={j} output={output} />
                  ))}
                </div>
              )}
            </>
          ) : (
            <pre className="nb-raw">{joinText(cell.source)}</pre>
          )}
        </div>
      ))}
    </div>
  )
}
