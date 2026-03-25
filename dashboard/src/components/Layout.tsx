import { Link, useLocation } from 'react-router-dom'
import type { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

const navItems = [
  { to: '/platform', label: 'Platform' },
  { to: '/agent', label: 'MAD Agent' },
  { to: '/thesis', label: 'Thesis' },
]

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const isThesis = location.pathname.startsWith('/thesis')

  const isActive = (path: string) => {
    return location.pathname.startsWith(path)
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: isThesis ? 'var(--paper)' : '#ffffff' }}>
      <header className="border-b" style={{ borderColor: isThesis ? 'var(--paper-deep)' : '#e5e7eb' }}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link
            to="/"
            className="text-xl font-semibold hover:opacity-70 transition-opacity"
            style={{
              fontFamily: 'var(--font-display)',
              color: isThesis ? 'var(--ink)' : '#111827',
            }}
          >
            Silon
          </Link>
          <nav className="flex gap-1">
            {navItems.map(({ to, label }) => (
              <Link
                key={to}
                to={to}
                className="px-4 py-2 rounded-md text-sm font-medium transition-colors"
                style={{
                  fontFamily: 'var(--font-display)',
                  color: isActive(to) ? 'var(--accent-strong)' : 'var(--ink-muted)',
                  backgroundColor: isActive(to) ? 'var(--accent-soft)' : 'transparent',
                }}
              >
                {label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main>
        {children}
      </main>
    </div>
  )
}
