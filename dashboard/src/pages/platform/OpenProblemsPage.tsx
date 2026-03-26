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

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Does veracity matter?
            </h2>
            <p className="leading-relaxed mb-4">
              There is a more fundamental question lurking beneath the design of this
              platform: does veracity&mdash;whether a conjecture is actually true&mdash;matter
              at all in science? At first glance, the answer seems obviously yes. Science is
              supposed to be about discovering truths about the natural world. But if you look
              carefully at how science actually works, and at what makes it useful, the
              relationship between scientific progress and truth is far less straightforward
              than it appears.
            </p>
            <p className="leading-relaxed mb-4">
              Consider Newtonian mechanics. For over two centuries, it was the most successful
              scientific theory in history. It predicted planetary orbits, tides, projectile
              trajectories, and the existence of Neptune before anyone had seen it. Then
              general relativity showed that Newton&rsquo;s framework was, strictly speaking,
              wrong&mdash;space is curved, gravity is not a force, and the equations only
              approximate reality in the low-energy limit. Was Newtonian mechanics not science?
              Were the two centuries of engineering, navigation, and astronomy built on it
              somehow illegitimate? Of course not. The theory was spectacularly useful despite
              not being veracious in the deepest sense.
            </p>
            <p className="leading-relaxed mb-4">
              This pattern repeats across the history of science. Ptolemaic epicycles predicted
              planetary positions well enough to navigate by. Caloric theory guided early
              thermodynamics. The plum pudding model of the atom made predictions that were
              partly confirmed before being replaced. In each case, the theory was
              &ldquo;wrong&rdquo; but scientifically productive. What mattered was not
              whether the conjecture corresponded to ultimate reality, but whether it
              generated predictions that held up under testing&mdash;whether it had, in the
              language of the previous section, predictive capacity.
            </p>
            <p className="leading-relaxed mb-4">
              If this is right&mdash;if what science actually tracks is predictive capacity
              rather than truth&mdash;then veracity may not be the thing this platform should
              be optimizing for. A conjecture market that prices claims based on how well they
              predict observations is already doing the work of science, regardless of whether
              the claims are &ldquo;true&rdquo; in some deeper metaphysical sense. The price
              of a conjecture reflects the weight of evidence, and the weight of evidence
              reflects predictive success, and predictive success is what lets us build bridges
              and cure diseases. Truth, if it enters at all, enters as a downstream consequence
              of sustained predictive capacity&mdash;not as a prerequisite for it.
            </p>

            <h3
              className="text-lg font-semibold mb-4 mt-8"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Why the platform might still matter even if veracity doesn&rsquo;t
            </h3>
            <p className="leading-relaxed mb-4">
              If veracity is not the central concern, one might ask: what is this platform
              even for? The answer is that the platform solves two problems that plague
              science regardless of your stance on truth.
            </p>
            <ul className="space-y-4 mb-6">
              <li className="leading-relaxed">
                <strong>A venue for null results.</strong> Current scientific publishing has a
                well-documented bias toward positive results. Experiments that fail to confirm
                a hypothesis are difficult to publish, which means the evidence base is
                systematically distorted. Researchers run the same failed experiment over and
                over because no one published the negative result. In a conjecture market,
                null results are not unpublishable&mdash;they are profitable. If you hold a
                position against a conjecture and the evidence comes back negative, the price
                drops and you gain. The market does not care whether a result is
                &ldquo;interesting&rdquo; in the way a journal editor does. It cares whether
                the result moves the price. This means null results are naturally absorbed
                into the evidence base, which improves the overall accuracy of the market
                whether or not we think accuracy is the same as truth.
              </li>
              <li className="leading-relaxed">
                <strong>A space for uncertain conjectures.</strong> Science advances not just
                by confirming well-formed hypotheses, but by entertaining speculative
                ideas&mdash;conjectures that might be wrong, that are half-formed, that
                connect two domains in a way no one has tested yet. In the current system,
                publishing a speculative conjecture is reputationally risky. If it turns out
                to be wrong, it can damage your career. In a conjecture market, you can float
                a speculative idea at a low price&mdash;say 0.15&mdash;which honestly
                communicates your uncertainty. If evidence later supports it, the price rises
                and you benefit. If it doesn&rsquo;t, you lose a small stake, but
                you&rsquo;ve still contributed the idea to the public record where others can
                build on it or rule it out. The market gives you a way to say &ldquo;I think
                this might be true but I&rsquo;m not sure&rdquo; and have that contribution
                valued appropriately.
              </li>
            </ul>
            <p className="leading-relaxed mb-4">
              Both of these benefits hold whether or not you believe science is in the business
              of discovering capital-T Truth. Even a strict instrumentalist&mdash;someone who
              thinks scientific theories are just tools for prediction, not descriptions of
              reality&mdash;should want a system where negative evidence is rewarded and
              speculative conjectures can be floated without career risk. The platform is not
              a truth machine. It is an evidence-aggregation machine. And
              evidence aggregation is valuable regardless of your metaphysics.
            </p>

            <div
              className="rounded-lg border p-6 mt-6"
              style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
            >
              <h3
                className="text-sm font-semibold uppercase tracking-widest mb-4"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                The deeper question
              </h3>
              <p className="leading-relaxed mb-4">
                This opens a philosophical tension that the platform does not resolve, and
                perhaps should not try to. If two conjectures make the same predictions in
                every testable scenario but disagree on unobservable ontology&mdash;if one says
                &ldquo;electrons are real particles&rdquo; and the other says &ldquo;electron
                is a useful fiction that compresses our observations&rdquo;&mdash;the market
                will price them identically. It has no way to distinguish them, because there
                is no evidence that could distinguish them. Whether this is a feature or a
                limitation depends on whether you think science should be in the business of
                adjudicating questions that have no empirical consequences. The market&rsquo;s
                implicit answer is no: if it doesn&rsquo;t cash out in predictions, it
                doesn&rsquo;t get a price.
              </p>
              <p className="leading-relaxed">
                This is not a settled question. But it is worth noting that the platform
                remains useful under either interpretation. If you think veracity matters, the
                market approximates it through evidence aggregation. If you think it
                doesn&rsquo;t, the market still does the useful work of surfacing predictive
                capacity, rewarding null results, and giving speculative conjectures a home.
                The question of whether truth is the goal or a byproduct is one the platform
                can afford to leave open.
              </p>
            </div>
          </section>

        </div>
      </div>
    </div>
  )
}
