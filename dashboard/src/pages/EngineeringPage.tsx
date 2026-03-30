import { Link } from 'react-router-dom'

const guides = [
  {
    slug: 'backtesting',
    title: 'Backtesting',
    description: 'Simulating 200 years of science to validate the market mechanism against historical data.',
  },
  {
    slug: 'market-making',
    title: 'Market Making API',
    description: 'The API for creating conjectures, trading shares, submitting evidence, and tracking portfolios.',
  },
]

export default function EngineeringPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-4xl mx-auto px-6 py-12">
        <header className="mb-16">
          <h1
            className="text-4xl font-bold mb-4 leading-tight"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
          >
            Engineering
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Infrastructure, APIs, and simulation: how the conjecture market is built and tested.
          </p>
        </header>

        <div className="space-y-4">
          {guides.map(({ slug, title, description }) => (
            <Link
              key={slug}
              to={`/engineering/${slug}`}
              className="block rounded-lg border p-6 hover:border-current transition-colors"
              style={{
                borderColor: 'var(--paper-deep)',
                color: 'var(--ink)',
              }}
            >
              <h2
                className="text-lg font-semibold mb-1"
                style={{ fontFamily: 'var(--font-display)' }}
              >
                {title}
              </h2>
              <p
                className="text-sm leading-relaxed"
                style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
              >
                {description}
              </p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
