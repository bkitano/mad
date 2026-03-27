import { Link } from 'react-router-dom'

export default function ShortingPage() {
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
            Shorting
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            What it means to take a position against a conjecture.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The basic idea
          </h2>

          <p className="leading-relaxed">
            In a conjecture market, every conjecture has a price between 0 and 1
            representing the community&rsquo;s credence that it is true. Buying a
            conjecture means you believe the price should be higher &mdash; you
            think the community is underestimating the likelihood that the claim
            is true. <strong>Shorting</strong> a conjecture means you believe the
            price should be lower &mdash; you think the community is
            overestimating it.
          </p>

          <p className="leading-relaxed">
            In traditional financial markets, shorting requires borrowing shares
            from someone who owns them, selling those shares, and later buying
            them back (hopefully at a lower price) to return them. This
            introduces counterparty risk, margin requirements, and the
            possibility of a short squeeze.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Shorting in the conjecture market
          </h2>

          <p className="leading-relaxed">
            The conjecture market does not have a fixed supply of shares. There
            is no finite pool of conjecture tokens that must be borrowed and
            returned. Instead, the market uses an automated pricing mechanism
            (see{' '}
            <Link
              to="/platform/price-determination"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Price Determination
            </Link>
            ) that adjusts the price in response to the weight of positions on
            each side.
          </p>

          <p className="leading-relaxed">
            Taking a short position means registering your belief that a
            conjecture is false, weighted by some measure of conviction. The
            effect on the market price depends on the pricing mechanism: your
            position pushes the price downward in proportion to how much
            conviction you express, counterbalanced by everyone else&rsquo;s
            positions.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            When shorting is informative
          </h2>

          <p className="leading-relaxed">
            Shorting is the market&rsquo;s mechanism for incorporating
            disconfirming evidence and skepticism. Without shorts, the market
            can only move prices upward &mdash; it can express growing
            confidence but not growing doubt. A market that only allows buying
            is systematically biased toward optimism.
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Provably false conjectures.</strong> When &ldquo;heavier
              objects fall faster in a vacuum&rdquo; is listed, informed
              participants short it and the price drops to ~0.01. The short
              positions are the signal that this conjecture is wrong.
            </li>
            <li className="leading-relaxed">
              <strong>Overconfident conjectures.</strong> If a conjecture is
              priced at 0.80 but the evidence only supports 0.50, a participant
              can short it and profit if the price corrects. This is how the
              market self-corrects against hype.
            </li>
            <li className="leading-relaxed">
              <strong>Escape-hatch conjectures.</strong> Attempting to short an
              unfalsifiable conjecture reveals its pathology &mdash; the short
              seller can never collect because the escape clause absorbs every
              piece of counter-evidence. The failure to profit from a short is
              itself diagnostic information about the conjecture&rsquo;s quality.
            </li>
          </ul>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            When shorting is problematic
          </h2>

          <p className="leading-relaxed">
            Shorting introduces the possibility of strategic suppression &mdash;
            a participant could short a conjecture not because they believe it
            is false, but to discourage others from investigating it. In
            financial markets, this is analogous to short-and-distort campaigns.
            The conjecture market needs mechanisms to distinguish legitimate
            skepticism from strategic manipulation:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Evidence requirements.</strong> A short position that
              attaches disconfirming evidence is more credible than a bare
              short. The market could weight evidenced shorts more heavily in
              the pricing mechanism.
            </li>
            <li className="leading-relaxed">
              <strong>Stake-proportional influence.</strong> If shorting
              requires staking conviction (see{' '}
              <Link
                to="/platform/price-determination"
                className="underline"
                style={{ color: 'var(--accent)' }}
              >
                Price Determination
              </Link>
              ), the cost of strategic suppression scales with the size of the
              position, making large-scale manipulation expensive.
            </li>
          </ul>

          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Shorting vs. not holding
            </h3>
            <p className="leading-relaxed">
              There is an important distinction between <em>not holding</em> a
              conjecture and <em>shorting</em> it. Not holding means you have no
              opinion or no stake &mdash; the conjecture&rsquo;s price movement
              doesn&rsquo;t affect you. Shorting means you are actively claiming
              the conjecture is overpriced and you want to profit from (or at
              least signal) its decline. The difference matters because the
              market needs to distinguish between absence of belief and presence
              of disbelief.
            </p>
          </div>

        </div>
      </div>
    </div>
  )
}
