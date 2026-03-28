import { Link } from 'react-router-dom'

const guides = [
  {
    slug: 'hello-world',
    title: 'Hello, World',
    description: 'Portfolios, positions, and portfolio veracity consensus.',
  },
  {
    slug: 'scoring-metrics',
    title: 'Scoring & Metrics',
    description: 'How portfolio value, rolling scores, attribution graphs, trade history, and conjecture impact are calculated.',
  },
  {
    slug: 'creating-conjectures',
    title: 'Creating Conjectures',
    description: 'How to create, price, refine, and manage conjectures in the market.',
  },
  {
    slug: 'example-conjectures',
    title: 'Example Conjectures',
    description: 'A catalog of good, bad, and misleading conjectures — with market analysis for each.',
  },
  {
    slug: 'market-incentives',
    title: 'Market Incentives',
    description: 'The behaviors we want the market to produce, and what they imply about pricing.',
  },
  {
    slug: 'price-determination',
    title: 'Price Determination',
    description: 'How the market computes the price of a conjecture from participant positions.',
  },
  {
    slug: 'shorting',
    title: 'Shorting',
    description: 'What it means to take a position against a conjecture.',
  },
  {
    slug: 'integrations',
    title: 'Integrations',
    description: 'Connecting the conjecture market to arXiv, Google Scholar, and peer-reviewed venues.',
  },
  {
    slug: 'bayesian-networks',
    title: 'Bayesian Networks',
    description: 'How conjecture dependencies emerge from correlated trades, and the challenge of extracting causal structure from market behavior.',
  },
  {
    slug: 'open-problems',
    title: 'Open Problems in Market Incentives and Alignment',
    description: 'Unsolved design questions: rewarding predictive capacity, not just specificity.',
  },
]

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

        <div className="space-y-4">
          {guides.map(({ slug, title, description }) => (
            <Link
              key={slug}
              to={`/platform/${slug}`}
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
