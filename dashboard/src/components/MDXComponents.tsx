import type { MDXComponents } from 'mdx/types'
import type { HTMLAttributes } from 'react'
import CodeBlock from './CodeBlock'
import * as mdxRegistry from './mdx'

function HeadingLink({ Tag, id, children, ...props }: { Tag: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' } & HTMLAttributes<HTMLHeadingElement>) {
  return (
    <Tag id={id} className="heading-anchor group" {...props}>
      {children}
      {id && (
        <a href={`#${id}`} className="heading-anchor-link" aria-label={`Link to ${typeof children === 'string' ? children : 'section'}`}>
          #
        </a>
      )}
    </Tag>
  )
}

export const mdxComponents: MDXComponents = {
  ...mdxRegistry,
  pre: CodeBlock,
  h1: (props) => <HeadingLink Tag="h1" {...props} />,
  h2: (props) => <HeadingLink Tag="h2" {...props} />,
  h3: (props) => <HeadingLink Tag="h3" {...props} />,
  h4: (props) => <HeadingLink Tag="h4" {...props} />,
  h5: (props) => <HeadingLink Tag="h5" {...props} />,
  h6: (props) => <HeadingLink Tag="h6" {...props} />,
}
