import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-gray-200">
        <div className="max-w-3xl mx-auto px-4 py-6 flex items-center justify-between">
          <Link to="/" className="text-xl font-semibold text-gray-900 hover:text-gray-600">
            😡 MAD
          </Link>
          <nav className="flex gap-6">
            <Link to="/notes" className="text-gray-600 hover:text-gray-900">
              Notes
            </Link>
          </nav>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 py-8">
        {children}
      </main>

      <footer className="border-t border-gray-200 mt-16">
        <div className="max-w-3xl mx-auto px-4 py-6 text-sm text-gray-500">
          Published with Vite
        </div>
      </footer>
    </div>
  )
}
