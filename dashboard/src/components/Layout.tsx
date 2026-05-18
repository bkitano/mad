import { Link, useLocation } from 'react-router-dom'
import { useState, useRef, useEffect } from 'react'
import type { ReactNode } from 'react'
import { useAuth } from '../contexts/AuthContext'

interface LayoutProps {
  children: ReactNode
}

const navItems = [
  { to: '/platform', label: 'Platform' },
  { to: '/engineering', label: 'Engineering' },
  { to: '/agent', label: 'MAD Agent' },
  { to: '/thesis', label: 'Thesis' },
]

function AccountMenu() {
  const { user, signOut, signInWithGoogle } = useAuth()
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [open])

  if (user) {
    const avatar = user.user_metadata?.avatar_url
    const name = user.user_metadata?.full_name || user.email
    const email = user.email

    return (
      <div className="relative" ref={menuRef}>
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-gray-100 transition-colors"
        >
          {avatar ? (
            <img src={avatar} alt="" className="w-7 h-7 rounded-full" />
          ) : (
            <div className="w-7 h-7 rounded-full bg-gray-200 flex items-center justify-center text-xs font-medium text-gray-600">
              {(name || '?')[0].toUpperCase()}
            </div>
          )}
          <svg className="w-3.5 h-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {open && (
          <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
            <div className="px-4 py-3 border-b border-gray-100">
              <p className="text-sm font-medium text-gray-900 truncate">{name}</p>
              {email && <p className="text-xs text-gray-500 truncate mt-0.5">{email}</p>}
            </div>
            <button
              onClick={() => { signOut(); setOpen(false) }}
              className="w-full text-left px-4 py-2.5 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Sign out
            </button>
          </div>
        )}
      </div>
    )
  }

  return (
    <button
      onClick={signInWithGoogle}
      className="px-3 py-1.5 rounded-md text-sm font-medium text-white bg-gray-900 hover:bg-gray-800 transition-colors"
    >
      Sign in
    </button>
  )
}

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
          <div className="flex items-center gap-4">
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
            <AccountMenu />
          </div>
        </div>
      </header>

      <main>
        {children}
      </main>
    </div>
  )
}
