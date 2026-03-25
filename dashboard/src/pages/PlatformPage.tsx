export default function PlatformPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-4xl mx-auto px-6 py-12">
        <header className="mb-16">
          <h1
            className="text-4xl font-bold mb-4 leading-tight"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
          >
            Platform
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Documentation, tutorials, and scenarios for how the Conjecture Market operates.
          </p>
        </header>

        <div
          className="rounded-lg border p-8 text-center"
          style={{
            borderColor: 'var(--paper-deep)',
            backgroundColor: 'var(--paper)',
            color: 'var(--ink-muted)',
            fontFamily: 'var(--font-body)',
          }}
        >
          <p className="text-lg mb-2">Coming soon.</p>
          <p className="text-sm">
            This section will contain walkthroughs, worked examples, and reference
            documentation for the conjecture market.
          </p>
        </div>
      </div>
    </div>
  )
}
