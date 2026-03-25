import { Link } from 'react-router-dom'

export default function OpenProblemsPage() {
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
            Open Problems in Market Incentives and Alignment
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Unsolved design questions that will shape how the conjecture market
            behaves in practice.
          </p>
        </header>

        <div className="space-y-12" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Rewarding predictive capacity, not just specificity
            </h2>
            <p className="leading-relaxed mb-4">
              The market naturally rewards conjectures that are specific and testable,
              because their prices move cleanly in response to evidence. But specificity
              alone is not what makes science valuable. A conjecture
              like &ldquo;this particular coin will land heads on the next flip&rdquo; is
              maximally specific and testable, but it has no predictive capacity beyond the
              single event it describes. It tells you nothing about the world.
            </p>
            <p className="leading-relaxed mb-4">
              The conjectures that matter most in science are the ones with broad predictive
              reach: they make correct predictions across many situations, not just one.
              Newton&rsquo;s law of gravitation was not valuable because it predicted one
              apple falling&mdash;it was valuable because it predicted the motion of every
              massive body, from cannonballs to moons. General relativity was not valuable
              because it predicted one eclipse measurement&mdash;it was valuable because the
              same framework predicted gravitational lensing, time dilation, frame dragging,
              and gravitational waves, across conditions that hadn&rsquo;t been tested yet.
            </p>
            <p className="leading-relaxed mb-4">
              This creates a design problem. The market, as described so far, rewards you
              for holding a conjecture whose price goes up. But a narrow conjecture that
              gets confirmed once and a broad theory that gets confirmed across dozens of
              experiments might produce the same price movement&mdash;both go from 0.4 to
              0.9. The narrow one is a lucky bet. The broad one is a scientific
              contribution. How does the market distinguish them?
            </p>
            <p className="leading-relaxed mb-4">
              Several approaches are worth exploring:
            </p>
            <ul className="space-y-4">
              <li className="leading-relaxed">
                <strong>Downstream conjecture creation.</strong> A conjecture with real
                predictive capacity tends to generate offspring&mdash;new, more specific
                conjectures derived from it. If a broad conjecture spawns many downstream
                conjectures that are themselves confirmed, that is evidence of predictive
                reach. The market could track the dependency graph and weight conjectures
                not just by their own price movement, but by the aggregate price movement
                of their descendants.
              </li>
              <li className="leading-relaxed">
                <strong>Cross-domain confirmation.</strong> A conjecture that is confirmed
                by evidence from multiple independent domains is more likely to reflect
                genuine predictive capacity than one confirmed by a single line of evidence.
                Darwin&rsquo;s theory was confirmed by biogeography, paleontology,
                embryology, and later genetics&mdash;each independent of the others. A
                market mechanism that recognizes when a conjecture&rsquo;s price is being
                moved by evidence from diverse sources could weight those price movements
                more heavily.
              </li>
              <li className="leading-relaxed">
                <strong>Prediction registration.</strong> Participants could register
                specific predictions derived from a conjecture before the evidence arrives.
                A conjecture that generates many successful registered predictions
                demonstrates predictive capacity in a way that is hard to fake.
                This is similar to pre-registration in clinical trials, but applied to the
                full scope of a conjecture&rsquo;s implications.
              </li>
              <li className="leading-relaxed">
                <strong>Compression as a signal.</strong> A conjecture that explains many
                observations with few assumptions is compressing the data&mdash;this is
                the essence of Occam&rsquo;s Razor and minimum description length. If two
                conjectures have the same price, but one is a single sentence and the other
                requires ten pages of caveats, the simpler one is doing more
                work per unit of specificity. The market might implicitly reflect this
                through liquidity: participants prefer to trade in the simpler conjecture
                because it is easier to reason about, which concentrates capital and
                attention on the more compressive claim.
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
                Historical example: The germ theory of disease
              </h3>
              <p className="leading-relaxed mb-4">
                In the 1860s, Pasteur and Koch advanced the conjecture that specific
                microorganisms cause specific diseases. This was not just a specific
                claim&mdash;it was a framework with enormous predictive reach. It predicted
                that cholera had a microbial cause (confirmed), that surgical infections
                could be reduced by antisepsis (confirmed by Lister), that rabies could be
                prevented by vaccination (confirmed), and that tuberculosis was caused by a
                specific bacterium (confirmed by Koch).
              </p>
              <p className="leading-relaxed">
                A narrow conjecture like &ldquo;this particular wound infection was caused
                by bacteria&rdquo; would have been confirmed and priced at 0.95. But the
                germ theory of disease would have generated dozens of downstream conjectures,
                each of which was independently confirmed. The aggregate price movement
                across that family of conjectures is what reveals the predictive
                capacity of the parent theory. The open problem is how to make this
                visible and rewardable in the market&rsquo;s incentive structure.
              </p>
            </div>

            <p className="leading-relaxed mt-6">
              This remains an open problem. The risk of not solving it is a market that
              optimizes for narrow, easily confirmed claims at the expense of the deep
              theories that actually advance scientific understanding. The risk of
              over-engineering a solution is a market that is too complex for participants
              to reason about, which undermines the simplicity that makes the core mechanism
              work.
            </p>
          </section>

        </div>
      </div>
    </div>
  )
}
