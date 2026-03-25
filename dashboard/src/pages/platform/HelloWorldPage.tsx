import { Link } from 'react-router-dom'

export default function HelloWorldPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-8">
          <Link
            to="/platform"
            className="text-sm hover:underline"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
          >
            &larr; Platform
          </Link>
        </div>

        <header className="mb-12">
          <h1
            className="text-3xl font-bold mb-4 leading-tight"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
          >
            Hello, World
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Portfolios, positions, and portfolio veracity consensus.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>
          <p className="leading-relaxed">
            Every participant in the conjecture market has a <strong>portfolio</strong>: a
            collection of positions in conjectures they have bought or sold. Each conjecture
            has a price reflecting the community&rsquo;s current credence in it&mdash;a number
            between 0 and 1 representing how likely the community believes it is to be true.
          </p>

          <p className="leading-relaxed">
            Your portfolio value is the sum of your positions, each weighted by the current
            price of the conjecture. As the community&rsquo;s beliefs shift&mdash;new evidence
            is published, results are reproduced or refuted&mdash;prices move, and your
            portfolio value moves with them.
          </p>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Example
            </h3>
            <p className="leading-relaxed mb-4">
              Suppose you are a researcher studying protein folding. You believe that a
              recent conjecture&mdash;&ldquo;AlphaFold-class models generalize to
              disordered proteins&rdquo;&mdash;is underpriced at 0.35. You buy a position.
            </p>
            <p className="leading-relaxed mb-4">
              Over the next few months, two independent groups publish results showing
              strong performance on disordered protein benchmarks. The price rises to 0.62.
              Your portfolio reflects that gain.
            </p>
            <p className="leading-relaxed">
              Conversely, if a replication attempt fails and the price drops to 0.20, your
              portfolio reflects that loss. The market doesn&rsquo;t care about your
              credentials or your intent&mdash;only whether the community&rsquo;s evolving
              consensus moves in the direction you positioned for.
            </p>
          </div>

          <p className="leading-relaxed">
            This is the fundamental unit of participation: you hold conjectures, and your
            portfolio value tracks how well your scientific judgment aligns with the evidence
            as it unfolds. We call this your <strong>portfolio veracity
            consensus</strong>&mdash;a single time series that captures, in aggregate, how
            good your bets on the truth have been.
          </p>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              What your portfolio tells you
            </h3>
            <ul className="space-y-3">
              <li className="leading-relaxed">
                <strong>Rising portfolio value</strong> means your positions are being
                validated by new evidence. The conjectures you believed in are gaining
                credence.
              </li>
              <li className="leading-relaxed">
                <strong>Falling portfolio value</strong> means the evidence is moving against
                your positions. Time to re-evaluate your beliefs, or double down if you think
                the market is wrong.
              </li>
              <li className="leading-relaxed">
                <strong>Flat portfolio value</strong> means the conjectures you hold are
                stable&mdash;no significant new evidence has shifted the community&rsquo;s
                beliefs in either direction.
              </li>
            </ul>
          </div>

          <p className="leading-relaxed">
            Over time, a consistently appreciating portfolio is the market&rsquo;s way of
            saying you have good scientific intuition. You identified which conjectures
            would be borne out before the evidence arrived. That track record is visible,
            verifiable, and independent of institutional affiliation or publication count.
          </p>
        </div>
      </div>
    </div>
  )
}
