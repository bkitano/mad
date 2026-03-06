import { isValidElement, type HTMLAttributes, type ReactNode } from 'react'
import Mermaid from './Mermaid'

type CodeBlockProps = HTMLAttributes<HTMLPreElement> & {
  children?: ReactNode
}

export default function CodeBlock({ children, ...props }: CodeBlockProps) {
  if (isValidElement(children)) {
    const codeProps = children.props as {
      className?: string
      children?: React.ReactNode
    }
    const className = codeProps.className
    const isMermaid = className?.includes('language-mermaid')

    if (isMermaid && typeof codeProps.children === 'string') {
      console.debug('[mdx] render mermaid block')
      return <Mermaid chart={codeProps.children.trim()} />
    }

    return <pre {...props}>{children}</pre>
  }

  return <pre {...props}>{children}</pre>
}
