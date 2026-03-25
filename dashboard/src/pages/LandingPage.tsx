import { Link } from 'react-router-dom'

export default function LandingPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-4xl mx-auto px-6 py-24">
        <div className="text-center mb-20">
          <h1
            className="text-6xl font-bold mb-6 leading-tight"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
          >
            Silon
          </h1>
          <p
            className="text-xl leading-relaxed max-w-2xl mx-auto"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Building the infrastructure for scientific consensus in the age of autonomous research.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Link
            to="/thesis"
            className="rounded-lg border p-6 hover:border-current transition-colors"
            style={{
              borderColor: 'var(--paper-deep)',
              color: 'var(--ink)',
              fontFamily: 'var(--font-display)',
            }}
          >
            <h2 className="text-lg font-semibold mb-2">Thesis</h2>
            <p
              className="text-sm leading-relaxed"
              style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
            >
              The position paper on scientific discovery, veracity, and market mechanisms
              in the age of AI.
            </p>
          </Link>

          <Link
            to="/platform"
            className="rounded-lg border p-6 hover:border-current transition-colors"
            style={{
              borderColor: 'var(--paper-deep)',
              color: 'var(--ink)',
              fontFamily: 'var(--font-display)',
            }}
          >
            <h2 className="text-lg font-semibold mb-2">Platform</h2>
            <p
              className="text-sm leading-relaxed"
              style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
            >
              Documentation, tutorials, and worked scenarios for how the conjecture
              market operates.
            </p>
          </Link>

          <Link
            to="/agent"
            className="rounded-lg border p-6 hover:border-current transition-colors"
            style={{
              borderColor: 'var(--paper-deep)',
              color: 'var(--ink)',
              fontFamily: 'var(--font-display)',
            }}
          >
            <h2 className="text-lg font-semibold mb-2">MAD Agent</h2>
            <p
              className="text-sm leading-relaxed"
              style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
            >
              The research agent dashboard for monitoring and running experiments.
            </p>
          </Link>
        </div>
      </div>
    </div>
  )
}
