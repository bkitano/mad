import { Link } from 'react-router-dom'
import { BlockMath, InlineMath } from 'react-katex'

export default function PriceDeterminationPage() {
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
            Price Determination
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            How the market computes the price of a conjecture from the positions
            of its participants.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The core problem
          </h2>

          <p className="leading-relaxed">
            A conjecture&rsquo;s price is a number between 0 and 1 representing
            the community&rsquo;s aggregate credence that the conjecture is
            true. The pricing mechanism must take the individual positions of
            all participants and produce this single number. The mechanism
            determines everything about how the market behaves: how sensitive
            the price is to new participants, how much influence any one
            participant can have, and whether the price is stable or volatile.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Automated market makers
          </h2>

          <p className="leading-relaxed">
            Unlike traditional financial markets where prices emerge from a
            continuous double auction (buyers and sellers posting limit orders),
            the conjecture market uses an <strong>automated market maker
            (AMM)</strong>. The AMM is an algorithm that always stands ready to
            accept positions on either side of a conjecture, adjusting the price
            after each transaction.
          </p>

          <p className="leading-relaxed">
            The canonical choice for prediction-style markets is the{' '}
            <strong>Logarithmic Market Scoring Rule (LMSR)</strong>, introduced
            by Robin Hanson. In LMSR, the market maker maintains a cost
            function over the current state of outstanding positions:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="C(q_{\text{yes}}, q_{\text{no}}) = b \ln\!\left(e^{q_{\text{yes}}/b} + e^{q_{\text{no}}/b}\right)" />
          </div>

          <p className="leading-relaxed">
            where <InlineMath math="q_{\text{yes}}" /> and{' '}
            <InlineMath math="q_{\text{no}}" /> are the total number of
            outstanding yes-shares and no-shares, and{' '}
            <InlineMath math="b" /> is a <strong>liquidity parameter</strong>{' '}
            that controls how sensitive the price is to new trades. The
            instantaneous price of a yes-share is:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="p = \frac{e^{q_{\text{yes}}/b}}{e^{q_{\text{yes}}/b} + e^{q_{\text{no}}/b}}" />
          </div>

          <p className="leading-relaxed">
            This is simply a softmax over the outstanding positions. The price
            is always between 0 and 1. When{' '}
            <InlineMath math="q_{\text{yes}} = q_{\text{no}}" />, the price is
            exactly 0.50. As more yes-shares are purchased, the price moves
            toward 1. As more no-shares are purchased, the price moves toward 0.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The liquidity parameter
          </h2>

          <p className="leading-relaxed">
            The parameter <InlineMath math="b" /> determines how much a single
            position moves the price. A small <InlineMath math="b" /> means the
            price is highly sensitive &mdash; a single participant can move it
            significantly. A large <InlineMath math="b" /> means the price is
            sluggish &mdash; it takes many participants to shift it. This
            creates a direct trade-off:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Small <InlineMath math="b" />:</strong> Early participants
              have outsized influence. The price is responsive to new
              information but vulnerable to manipulation by a single actor.
            </li>
            <li className="leading-relaxed">
              <strong>Large <InlineMath math="b" />:</strong> The price is
              stable and resistant to manipulation, but slow to incorporate new
              information. It takes many participants to move the needle.
            </li>
          </ul>

          <p className="leading-relaxed">
            The right choice of <InlineMath math="b" /> may vary by conjecture.
            A new, speculative conjecture might benefit from a small{' '}
            <InlineMath math="b" /> so that early evidence moves the price
            quickly. A well-established conjecture with thousands of
            participants might need a large <InlineMath math="b" /> so that the
            price reflects broad consensus rather than individual noise.
          </p>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Example: how the price moves
            </h3>
            <p className="leading-relaxed mb-4">
              Consider a conjecture with <InlineMath math="b = 100" />,
              currently at <InlineMath math="q_{\text{yes}} = q_{\text{no}} = 0" />{' '}
              (price = 0.50). A participant buys 10 yes-shares. The new price
              is:
            </p>
            <div className="overflow-x-auto">
              <BlockMath math="p = \frac{e^{10/100}}{e^{10/100} + e^{0/100}} = \frac{e^{0.1}}{e^{0.1} + 1} \approx 0.525" />
            </div>
            <p className="leading-relaxed">
              A modest move. Now suppose 100 participants each buy 10
              yes-shares (<InlineMath math="q_{\text{yes}} = 1000" />):
            </p>
            <div className="overflow-x-auto">
              <BlockMath math="p = \frac{e^{1000/100}}{e^{1000/100} + 1} = \frac{e^{10}}{e^{10} + 1} \approx 0.99995" />
            </div>
            <p className="leading-relaxed">
              The price is now essentially 1.0. The LMSR naturally produces
              asymptotic behavior: it becomes exponentially harder to push the
              price from 0.99 to 1.00 than from 0.50 to 0.51.
            </p>
          </div>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Cost of a position
          </h2>

          <p className="leading-relaxed">
            In the standard LMSR, the cost of purchasing{' '}
            <InlineMath math="\Delta q" /> additional yes-shares when the market
            is in state <InlineMath math="(q_{\text{yes}}, q_{\text{no}})" /> is:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="\text{Cost} = C(q_{\text{yes}} + \Delta q,\, q_{\text{no}}) - C(q_{\text{yes}},\, q_{\text{no}})" />
          </div>

          <p className="leading-relaxed">
            The cost is denominated in whatever units the market uses as its
            medium of exchange. In the conjecture market, the question of what
            these units are &mdash; dollars, reputation points, staked
            conviction &mdash; is itself an{' '}
            <Link
              to="/platform/open-problems"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              open design problem
            </Link>
            . The pricing mechanism is agnostic to the unit; it only requires
            that positions have some cost to prevent unbounded manipulation.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Alternatives to LMSR
          </h2>

          <p className="leading-relaxed">
            LMSR is not the only option. Other mechanisms worth considering:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Constant-product AMM (Uniswap-style).</strong> Used in
              decentralized finance. The invariant{' '}
              <InlineMath math="q_{\text{yes}} \times q_{\text{no}} = k" />{' '}
              produces a different price curve with different sensitivity
              properties. Less well-studied for prediction markets.
            </li>
            <li className="leading-relaxed">
              <strong>Continuous double auction.</strong> Participants post
              limit orders (buy at 0.40, sell at 0.60) and the market matches
              them. More expressive but requires sufficient liquidity to
              function. Can produce wide spreads in thin markets.
            </li>
            <li className="leading-relaxed">
              <strong>Weighted opinion aggregation.</strong> Rather than a
              trading mechanism, simply compute a weighted average of
              participant credences, where weights reflect some measure of
              expertise or stake. Simpler, but loses the information-revealing
              properties of a market mechanism.
            </li>
          </ul>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            What the price means
          </h2>

          <p className="leading-relaxed">
            Regardless of the mechanism, the price of a conjecture at time{' '}
            <InlineMath math="t" /> is the community&rsquo;s aggregate credence
            that the conjecture is true, as revealed by participants&rsquo;
            willingness to stake conviction on it. A price of 0.70 means the
            community, weighted by the stakes of its participants, believes
            there is roughly a 70% chance the conjecture is true.
          </p>

          <p className="leading-relaxed">
            The price is not a vote. It is not an average of opinions. It is
            the output of a mechanism that gives more weight to participants
            who stake more, and that automatically adjusts as new participants
            enter or existing participants update their positions. The
            mechanism creates incentives for honest revelation: if you believe
            the true credence is 0.80 and the price is 0.60, you can profit
            by buying &mdash; and in doing so, you push the price closer to
            what you believe is correct.
          </p>

          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Key properties of the pricing mechanism
            </h3>
            <ul className="space-y-3">
              <li className="leading-relaxed">
                <strong>Bounded.</strong> The price is always in [0, 1],
                interpretable as a probability.
              </li>
              <li className="leading-relaxed">
                <strong>Path-independent.</strong> The final price depends only
                on the total outstanding positions, not the order in which they
                were placed.
              </li>
              <li className="leading-relaxed">
                <strong>Incentive-compatible.</strong> A participant who
                believes the true credence is{' '}
                <InlineMath math="p^*" /> maximizes their expected payoff by
                moving the price toward{' '}
                <InlineMath math="p^*" />.
              </li>
              <li className="leading-relaxed">
                <strong>Bounded loss.</strong> The market maker&rsquo;s maximum
                loss is <InlineMath math="b \ln 2" />, controlled by the
                liquidity parameter. This is the subsidy the system pays to
                keep the market functional.
              </li>
            </ul>
          </div>

        </div>
      </div>
    </div>
  )
}
