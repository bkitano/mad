declare module '*.md' {
  import type { ComponentType } from 'react'
  import type { MDXComponents } from 'mdx/types'
  const component: ComponentType<{ components?: MDXComponents }>
  export default component
  export const frontmatter: Record<string, unknown>
}
