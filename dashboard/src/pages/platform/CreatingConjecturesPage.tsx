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

          {/* Principles of good conjectures */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Principles of good conjectures
            </h2>
            <p className="leading-relaxed mb-4">
              Not all conjectures are created equal. Beyond being falsifiable and
              well-scoped, a good conjecture tends to exhibit several properties that
              emerge naturally from how the market operates.
            </p>

            <ul className="space-y-6">
              <li className="leading-relaxed">
                <strong>High trade volume.</strong> A conjecture that attracts a lot of
                trades is one that many participants have opinions about&mdash;because their
                own work is downstream of it. Trade volume is the market&rsquo;s measure of
                how many people care about a claim, which is a direct proxy for its
                impact. This is analogous to a paper&rsquo;s citation count: a highly
                cited paper is one that many subsequent results depend on. A
                high-volume conjecture is one that many portfolios are exposed to.
                The difference is that trade volume is a live, continuous signal, while
                citation counts accumulate slowly and are easy to game.
              </li>
              <li className="leading-relaxed">
                <strong>Entropy sensitivity to evidence.</strong> A good conjecture&rsquo;s
                credence moves when relevant evidence is published, which means its
                entropy changes. If a conjecture&rsquo;s entropy is stable not because
                the question is settled but because no one can figure out what evidence
                would be relevant, the conjecture is probably too vague. Conjectures
                whose credences respond crisply to new results are well-scoped by
                definition.
              </li>
              <li className="leading-relaxed">
                <strong>Downstream conjecture generation.</strong> The best conjectures
                are generative&mdash;they inspire more specific conjectures that test
                particular implications. A conjecture with many active descendants in the
                market is doing real theoretical work: it is compressing a family of
                specific predictions into a single claim. This is the market&rsquo;s
                version of a theory.
              </li>
              <li className="leading-relaxed">
                <strong>Narrow bid-ask spread.</strong> When the spread between what buyers
                are willing to pay and what sellers are willing to accept is tight, it means
                the market has a clear, shared understanding of what the conjecture means
                and what evidence is relevant. A wide spread signals ambiguity&mdash;participants
                disagree not about the truth of the claim, but about what the claim
                even means. Good conjectures have tight spreads.
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
                Historical example: The Higgs boson
              </h3>
              <p className="leading-relaxed mb-4">
                The conjecture &ldquo;there exists a scalar boson that gives mass to
                elementary particles via spontaneous symmetry breaking&rdquo; (Higgs, 1964)
                would have exhibited all four properties. It would have had enormous trade
                volume: virtually every particle physicist had work downstream of
                it. Its entropy would have been sensitive to evidence from each generation
                of colliders &mdash; each new result either resolving or deepening the
                uncertainty. It generated many downstream conjectures about the
                boson&rsquo;s mass, decay channels, and coupling constants. And the spread
                would have been tight, because the claim was precise enough that the
                community agreed on what would count as confirmation.
              </p>
              <p className="leading-relaxed">
                Compare this to a vague conjecture like &ldquo;there are undiscovered
                fundamental particles.&rdquo; This is almost certainly true, but it would
                have low trade volume (what would you do with a position in it?), low
                entropy sensitivity (what evidence would move the credence?), few
                downstream conjectures (it&rsquo;s too vague to derive specific
                predictions from), and a wide spread (participants would disagree on
                what counts).
              </p>
            </div>
          </section>

          {/* How new conjectures enter the market */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              How new conjectures enter the market
            </h2>
            <p className="leading-relaxed mb-4">
              When you create a conjecture, it enters the market with no
              shares outstanding. Credence starts at 0.50 &mdash; the
              market has no information, so uncertainty is at maximum.
              Entropy is at its peak, which means the cost of taking a
              position is at its highest. You are buying into pure,
              unresolved doubt.
            </p>
            <p className="leading-relaxed mb-4">
              The creator typically takes the first position, buying YES or
              NO to express their belief. This first trade moves the
              credence away from 0.50 and begins to reduce entropy. Other
              participants can then trade based on whether they agree or
              disagree, and the credence moves toward the community&rsquo;s
              collective estimate.
            </p>
            <p className="leading-relaxed mb-4">
              Because the cost of a position is entropy-derived, early
              participation in a new conjecture is expensive &mdash; you
              are paying for maximum uncertainty. But the potential reward
              is also highest: if the credence eventually moves far in your
              direction, you bought at peak entropy and rode the full
              directional movement. This is the market&rsquo;s way of
              rewarding people who engage with new ideas when nobody else
              has.
            </p>
            <p className="leading-relaxed mb-4">
              A conjecture with no positions is just a statement. The act
              of taking the first position &mdash; of paying the entropy
              cost &mdash; is what turns it into a live instrument that the
              community can engage with.
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
                In 1982, Barry Marshall and Robin Warren proposed that
                stomach ulcers were caused by a bacterium, not stress or
                diet. If Marshall had created the conjecture &ldquo;the
                majority of gastric ulcers are caused by H. pylori
                infection,&rdquo; it would have entered at credence 0.50
                with maximum entropy. Marshall buys YES &mdash; paying the
                highest possible cost, because uncertainty is total.
              </p>
              <p className="leading-relaxed">
                The establishment disagrees and buys NO, pushing the
                credence down toward 0.25. Both sides paid high entropy
                costs. As Marshall submits evidence (culturing the
                bacterium, the self-experimentation, clinical trials), the
                credence climbs toward 0.95. Marshall&rsquo;s reward is
                large: he entered at peak entropy and the consensus moved
                far toward his position. The establishment doctors who
                bought NO at the same high entropy lose their stake &mdash;
                they bore the same uncertainty in the wrong direction.
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
              When two conjectures are very similar, their credences should be correlated.
              If they diverge, that divergence is a signal: either the market sees a
              meaningful distinction between them, or one is mispriced. Participants who
              notice the discrepancy can arbitrage it by buying the underpriced one and
              selling the overpriced one, which pushes the credences back toward
              consistency.
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
                conjectures would have settled at moderate credences with low remaining
                entropy, while a new, more precise conjecture &mdash; &ldquo;light exhibits
                wave-particle duality, behaving as a wave in propagation and as a particle
                in interaction with matter&rdquo; &mdash; would have attracted the
                liquidity. The new conjecture&rsquo;s high entropy (genuinely unresolved
                uncertainty) is where the reward potential lives. The old conjectures
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
              Refinement creates value because precise conjectures are more
              liquid &mdash; they are easier to evaluate, which means
              evidence moves their credence more efficiently and their
              entropy responds crisply to new results. A participant who
              creates a well-scoped refinement of a vague conjecture is
              providing the market with a better instrument for expressing
              and aggregating belief.
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
              conjecture will be more responsive to evidence &mdash; its credence will move
              more cleanly when relevant results are published, and its entropy will change
              more meaningfully, because the claim is specific enough for the market to
              interpret the evidence clearly.
            </p>
            <p className="leading-relaxed mb-4">
              In practice, this means selling your position in the broad conjecture and
              buying into the refined one. You are not abandoning your belief &mdash; you are
              expressing the same belief through a sharper instrument. If the refined
              conjecture still has high entropy (it is new and unresolved), you are entering
              at a cost that reflects genuine uncertainty, with the full potential for
              directional reward if the credence moves your way.
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
                atoms&rdquo; stayed true, but its credence was already near 1.0 and its
                entropy near zero &mdash; positions were cheap but there was no directional
                reward left. The refined conjectures were where the action was: high
                entropy, responsive to new experiments, and offering real returns for
                participants who bore that uncertainty in the right direction.
              </p>
            </div>
          </section>

        </div>
      </div>
    </div>
  )
}
