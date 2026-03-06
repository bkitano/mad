import type { ReactNode } from 'react'

interface CalloutProps {
  type?: 'info' | 'warning' | 'error' | 'success'
  title?: string
  children: ReactNode
}

const styles = {
  info: 'bg-blue-50 border-blue-200 text-blue-800',
  warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
  error: 'bg-red-50 border-red-200 text-red-800',
  success: 'bg-green-50 border-green-200 text-green-800',
}

export function Callout({ type = 'info', title, children }: CalloutProps) {
  return (
    <div className={`my-4 p-4 border-l-4 rounded-r ${styles[type]}`}>
      {title && <div className="font-semibold mb-1">{title}</div>}
      <div>{children}</div>
    </div>
  )
}
