import { Link } from 'react-router-dom'

export default function CreatingConjecturesPage() {
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
            Creating Conjectures
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            How to create, price, refine, and manage conjectures in the market.
          </p>
        </header>

        <div className="space-y-12" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* What is a conjecture */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              What is a conjecture?
            </h2>
            <p className="leading-relaxed mb-4">
              A conjecture is a falsifiable claim about the world, stated precisely enough
              that future evidence could move the community&rsquo;s credence in it. It is
              the atomic unit of the market. Good conjectures are specific, testable, and
              scoped&mdash;they say one thing, and they say it clearly.
            </p>
            <div
              className="rounded-lg border p-6"
              style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
            >
              <h3
                className="text-sm font-semibold uppercase tracking-widest mb-4"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                Examples
              </h3>
              <ul className="space-y-3">
                <li className="leading-relaxed">
                  <strong>Good:</strong> &ldquo;Transformer models with fewer than 1B
                  parameters can achieve &gt;90% accuracy on MATH benchmark problems
                  when trained with verifier-guided search.&rdquo;
                </li>
                <li className="leading-relaxed">
                  <strong>Too vague:</strong> &ldquo;Small models can do math.&rdquo;
                  &mdash; What counts as small? What counts as doing math? This isn&rsquo;t
                  testable in any precise sense.
                </li>
                <li className="leading-relaxed">
                  <strong>Too broad:</strong> &ldquo;Deep learning will solve
                  science.&rdquo; &mdash; No evidence could meaningfully update this.
                </li>
              </ul>

              <hr className="my-6" style={{ borderColor: 'var(--paper-deep)' }} />

              <h3
                className="text-sm font-semibold uppercase tracking-widest mb-4"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                Historical examples
              </h3>
              <ul className="space-y-3">
                <li className="leading-relaxed">
                  <strong>Good conjecture (Einstein, 1905):</strong> &ldquo;Light striking
                  a metal surface ejects electrons whose maximum kinetic energy depends
                  linearly on the frequency of the light and is independent of its
                  intensity.&rdquo; &mdash; Specific, testable, and Millikan confirmed it
                  experimentally in 1916.
                </li>
                <li className="leading-relaxed">
                  <strong>Good conjecture (Darwin, 1859):</strong> &ldquo;Species on
                  oceanic islands more closely resemble species on the nearest mainland
                  than species on other islands at similar latitudes.&rdquo; &mdash; A
                  specific, falsifiable prediction of common descent. Every biogeographic
                  survey since has been evidence for or against it.
                </li>
                <li className="leading-relaxed">
                  <strong>Too vague (historical):</strong> &ldquo;There is a vital force
                  that animates living matter.&rdquo; &mdash; Vitalism persisted for
                  centuries precisely because it was too vague to falsify. No specific
                  experiment could move its price.
                </li>
              </ul>
            </div>
          </section>

          {/* How conjectures are initially priced */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              How conjectures are initially priced
            </h2>
            <p className="leading-relaxed mb-4">
              When you create a conjecture, you set its initial price by taking the first
              position yourself. This initial price is your stated credence&mdash;your
              belief in the likelihood that the conjecture is true. By putting capital
              behind it, you are making that belief concrete.
            </p>
            <p className="leading-relaxed mb-4">
              The initial price serves as an anchor for the market. If you price a
              conjecture at 0.70, you are saying: &ldquo;I believe this is more likely true
              than not, and I&rsquo;m willing to risk capital on that belief.&rdquo; Other
              participants can then buy or sell based on whether they think your price is
              too high or too low, and the market moves toward the community&rsquo;s
              collective estimate.
            </p>
            <p className="leading-relaxed mb-4">
              A conjecture with no initial position is just a statement. The act of pricing
              it&mdash;of committing capital&mdash;is what turns it into a tradeable claim
              that the community can engage with.
            </p>

            <div
              className="rounded-lg border p-6"
              style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
            >
              <h3
                className="text-sm font-semibold uppercase tracking-widest mb-4"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                Historical example: Helicobacter pylori
              </h3>
              <p className="leading-relaxed mb-4">
                In 1982, Barry Marshall and Robin Warren proposed that stomach ulcers were
                caused by a bacterium, not stress or diet. The medical establishment was
                deeply skeptical. If Marshall had created a conjecture&mdash;&ldquo;the
                majority of gastric ulcers are caused by H. pylori
                infection&rdquo;&mdash;he might have priced it at 0.80, reflecting his
                strong conviction from his own culture results.
              </p>
              <p className="leading-relaxed">
                The rest of the market would have sold it down to perhaps 0.10, reflecting
                the overwhelming consensus that ulcers were a lifestyle disease. That gap
                between Marshall&rsquo;s price and the market&rsquo;s price is exactly the
                opportunity: when he famously drank a petri dish of H. pylori and developed
                gastritis, and when subsequent clinical trials confirmed antibiotic cures,
                the price would have climbed steadily toward 0.95. The initial price was
                wrong by the market&rsquo;s lights&mdash;but Marshall was right, and the
                market eventually reflected that.
              </p>
            </div>
          </section>

          {/* Dealing with similar conjectures */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Similar conjectures
            </h2>
            <p className="leading-relaxed mb-4">
              It is natural for multiple conjectures to overlap. Two researchers may
              independently create conjectures that say roughly the same thing, or a new
              conjecture may partially subsume an existing one. This is not a problem to
              be avoided&mdash;it is information.
            </p>
            <p className="leading-relaxed mb-4">
              When two conjectures are very similar, their prices should be correlated.
              If they diverge, that divergence is a signal: either the market sees a
              meaningful distinction between them, or one is mispriced. Participants who
              notice the discrepancy can arbitrage it by buying the underpriced one and
              selling the overpriced one, which pushes the prices back toward consistency.
            </p>
            <p className="leading-relaxed mb-4">
              Over time, the market naturally consolidates around the most precisely stated
              versions of an idea. Vague conjectures lose liquidity as participants migrate
              to sharper alternatives that are easier to evaluate and trade. You don&rsquo;t
              need a central authority to deduplicate&mdash;the market does it through
              attention and capital flow.
            </p>

            <div
              className="rounded-lg border p-6"
              style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
            >
              <h3
                className="text-sm font-semibold uppercase tracking-widest mb-4"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                Historical example: Competing models of light
              </h3>
              <p className="leading-relaxed mb-4">
                For centuries, &ldquo;light is a wave&rdquo; and &ldquo;light is a stream
                of particles&rdquo; coexisted as competing conjectures. Newton backed
                particles; Huygens backed waves. Young&rsquo;s double-slit experiment (1801)
                would have crashed the particle conjecture and surged the wave conjecture.
              </p>
              <p className="leading-relaxed">
                But then the photoelectric effect (1905) and quantum mechanics showed that
                both were partially right. The market would have reflected this: both
                conjectures would have settled at moderate prices, while a new, more precise
                conjecture&mdash;&ldquo;light exhibits wave-particle duality, behaving as
                a wave in propagation and as a particle in interaction with
                matter&rdquo;&mdash;would have attracted the liquidity. The old conjectures
                wouldn&rsquo;t disappear, but the capital would flow to the sharper one.
              </p>
            </div>
          </section>

          {/* Conjecture refinement */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Conjecture refinement
            </h2>
            <p className="leading-relaxed mb-4">
              Refinement is the process of replacing a conjecture with a more precise one.
              A refined conjecture narrows the scope, sharpens the prediction, or
              decomposes a broad claim into specific sub-claims. This is one of the most
              valuable things a participant can do in the market.
            </p>

            <div
              className="rounded-lg border p-6 mb-6"
              style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
            >
              <h3
                className="text-sm font-semibold uppercase tracking-widest mb-4"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                Historical example: Gravity
              </h3>
              <p className="leading-relaxed mb-3">
                <strong>Original (Newton, 1687):</strong> &ldquo;Every body attracts every
                other body with a force proportional to the product of their masses and
                inversely proportional to the square of the distance between them.&rdquo;
              </p>
              <p className="leading-relaxed mb-3">
                <strong>Refined (Einstein, 1915):</strong> &ldquo;Massive objects curve
                spacetime, and the trajectory of objects through curved spacetime produces
                the effects we observe as gravity. Newtonian gravity is the low-mass,
                low-velocity approximation.&rdquo;
              </p>
              <p className="leading-relaxed" style={{ color: 'var(--ink-muted)' }}>
                Einstein&rsquo;s conjecture did not invalidate Newton&rsquo;s&mdash;it
                refined it. It specified the conditions under which Newton&rsquo;s
                predictions break down (strong gravitational fields, high velocities) and
                made new, testable predictions (light bending around the sun, gravitational
                time dilation). The refined version was more precise and thus more
                liquid&mdash;Eddington&rsquo;s 1919 eclipse observation could move its
                price directly.
              </p>
            </div>

            <p className="leading-relaxed mb-4">
              Refinement creates value because precise conjectures are more liquid&mdash;they
              are easier to evaluate, which means evidence moves their price more efficiently.
              A participant who creates a well-scoped refinement of a vague conjecture is
              providing the market with a better instrument for expressing and aggregating
              belief.
            </p>
          </section>

          {/* What to do when you find a better conjecture */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              When you find a more precise conjecture than one you hold
            </h2>
            <p className="leading-relaxed mb-4">
              If you hold a position in a broad conjecture and discover that a more precise
              version exists, you should consider migrating your position. The refined
              conjecture will be more responsive to evidence&mdash;its price will move more
              cleanly when relevant results are published, because the claim is specific
              enough for the market to interpret the evidence clearly.
            </p>
            <p className="leading-relaxed mb-4">
              In practice, this means selling your position in the broad conjecture and
              buying into the refined one. You are not abandoning your belief&mdash;you are
              expressing the same belief through a sharper instrument. If you were right
              about the broad claim, you should be right about the specific one too, and you
              will capture the price movement more directly.
            </p>
            <p className="leading-relaxed mb-4">
              The exception is when you hold a broad conjecture precisely because you believe
              in the general principle, not any specific instantiation of it. In that case,
              holding the broad conjecture is itself a position&mdash;a bet that the idea is
              robust across conditions, not just in the specific scenario the refined version
              describes.
            </p>

            <div
              className="rounded-lg border p-6"
              style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
            >
              <h3
                className="text-sm font-semibold uppercase tracking-widest mb-4"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                Historical example: Atomic theory
              </h3>
              <p className="leading-relaxed mb-4">
                A 19th-century chemist might have held a position in &ldquo;matter is
                composed of indivisible atoms.&rdquo; By the early 1900s, more precise
                conjectures emerged: &ldquo;atoms contain a dense positive nucleus
                surrounded by electrons&rdquo; (Rutherford, 1911), then &ldquo;electrons
                occupy quantized energy levels&rdquo; (Bohr, 1913), then the full quantum
                mechanical model.
              </p>
              <p className="leading-relaxed">
                At each stage, the smart move would have been to migrate from the broad
                conjecture to the refined one. The broad claim &ldquo;matter is made of
                atoms&rdquo; stayed true, but its price was already near 1.0&mdash;there
                was no upside left. The refined conjectures were where the action was:
                their prices were volatile, responsive to new experiments, and offered
                real returns for participants who positioned correctly.
              </p>
            </div>
          </section>

        </div>
      </div>
    </div>
  )
}
