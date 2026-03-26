import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'

interface Section {
  id: string
  title: string
  level: number
}

const sections: Section[] = [
  { id: 'predictive-capacity', title: 'Rewarding predictive capacity, not just specificity', level: 1 },
  { id: 'downstream-creation', title: 'Downstream conjecture creation', level: 2 },
  { id: 'cross-domain', title: 'Cross-domain confirmation', level: 2 },
  { id: 'prediction-registration', title: 'Prediction registration', level: 2 },
  { id: 'compression', title: 'Compression as a signal', level: 2 },
  { id: 'does-veracity-matter', title: 'Does veracity matter?', level: 1 },
  { id: 'science-without-reproducibility', title: 'How science works without reproducibility', level: 2 },
  { id: 'platform-still-matters', title: 'Why the platform might still matter', level: 2 },
]

export default function OpenProblemsPage() {
  const [activeSection, setActiveSection] = useState(sections[0].id)

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id)
          }
        }
      },
      { rootMargin: '-80px 0px -60% 0px' }
    )

    sections.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [])

  const scrollTo = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-6xl mx-auto px-6 py-12 flex gap-12">

        {/* Side Table of Contents */}
        <nav className="hidden lg:block w-64 flex-shrink-0">
          <div className="sticky top-24">
            <div className="mb-6">
              <Link
                to="/platform"
                className="text-sm hover:underline"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
              >
                &larr; Platform
              </Link>
            </div>
            <h3
              className="text-xs font-semibold uppercase tracking-widest mb-6"
              style={{ color: 'var(--ink-muted)', fontFamily: 'var(--font-display)' }}
            >
              Table of Contents
            </h3>
            <ul className="space-y-1">
              {sections.map(({ id, title, level }) => (
                <li key={id}>
                  <button
                    onClick={() => scrollTo(id)}
                    className="block w-full text-left transition-colors duration-150 rounded px-2 py-1.5 text-sm"
                    style={{
                      paddingLeft: level === 2 ? '1.25rem' : '0.5rem',
                      fontFamily: 'var(--font-display)',
                      fontWeight: level === 1 ? 500 : 400,
                      fontSize: level === 1 ? '0.875rem' : '0.8125rem',
                      color: activeSection === id ? 'var(--accent-strong)' : 'var(--ink-muted)',
                      backgroundColor: activeSection === id ? 'var(--accent-soft)' : 'transparent',
                    }}
                  >
                    {title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        {/* Main Content */}
        <article className="flex-1 min-w-0">
          {/* Back link for mobile (hidden on lg where TOC has it) */}
          <div className="mb-8 lg:hidden">
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
                id="predictive-capacity"
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
                <li id="downstream-creation" className="leading-relaxed">
                  <strong>Downstream conjecture creation.</strong> A conjecture with real
                  predictive capacity tends to generate offspring&mdash;new, more specific
                  conjectures derived from it. If a broad conjecture spawns many downstream
                  conjectures that are themselves confirmed, that is evidence of predictive
                  reach. The market could track the dependency graph and weight conjectures
                  not just by their own price movement, but by the aggregate price movement
                  of their descendants.
                </li>
                <li id="cross-domain" className="leading-relaxed">
                  <strong>Cross-domain confirmation.</strong> A conjecture that is confirmed
                  by evidence from multiple independent domains is more likely to reflect
                  genuine predictive capacity than one confirmed by a single line of evidence.
                  Darwin&rsquo;s theory was confirmed by biogeography, paleontology,
                  embryology, and later genetics&mdash;each independent of the others. A
                  market mechanism that recognizes when a conjecture&rsquo;s price is being
                  moved by evidence from diverse sources could weight those price movements
                  more heavily.
                </li>
                <li id="prediction-registration" className="leading-relaxed">
                  <strong>Prediction registration.</strong> Participants could register
                  specific predictions derived from a conjecture before the evidence arrives.
                  A conjecture that generates many successful registered predictions
                  demonstrates predictive capacity in a way that is hard to fake.
                  This is similar to pre-registration in clinical trials, but applied to the
                  full scope of a conjecture&rsquo;s implications.
                </li>
                <li id="compression" className="leading-relaxed">
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
                id="does-veracity-matter"
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
                id="science-without-reproducibility"
                className="text-lg font-semibold mb-4 mt-8"
                style={{ fontFamily: 'var(--font-display)' }}
              >
                How science works without simple reproducibility
              </h3>
              <p className="leading-relaxed mb-4">
                This is not just a philosophical argument. There is an empirical question
                underneath it: does science actually work despite well-documented failures
                of reproducibility? And the answer, awkwardly, seems to be yes.
              </p>
              <p className="leading-relaxed mb-4">
                The reproducibility crisis is real and severe. The Open Science
                Collaboration&rsquo;s 2015 effort to replicate 100 psychology studies found
                that only 36% produced statistically significant results the second time
                around. Cancer biology has fared similarly&mdash;an Amgen team reported in 2012
                that they could reproduce only 6 of 53 landmark oncology papers. Surveys of
                researchers themselves confirm the pattern: a 2016 Nature survey found that
                over 70% of scientists had tried and failed to reproduce another
                scientist&rsquo;s work, and over half had failed to reproduce their own.
              </p>
              <p className="leading-relaxed mb-4">
                And yet. Drug development continues to produce effective therapies. The mRNA
                vaccines developed during COVID-19 worked, built on decades of immunology and
                molecular biology that included plenty of irreproducible individual results.
                CRISPR gene editing works. Semiconductor fabrication at 3-nanometer nodes works.
                SpaceX lands rockets on barges. The bridges hold. If the scientific literature
                were as broken as the reproducibility numbers suggest, you would expect the
                engineering built on top of it to fail much more often than it does.
              </p>
              <p className="leading-relaxed mb-4">
                So what explains the gap? How does science produce reliable technology and
                genuine understanding while resting on a foundation where individual results
                frequently don&rsquo;t replicate? Several mechanisms seem to be doing the
                real load-bearing work:
              </p>
              <ul className="space-y-4 mb-6">
                <li className="leading-relaxed">
                  <strong>Convergence across methods.</strong> Science rarely depends on any
                  single experiment. A finding becomes load-bearing when it is confirmed by
                  multiple independent methods, each with different biases and failure modes.
                  The structure of DNA was not established by one X-ray diffraction
                  image&mdash;it was confirmed by chemical analysis, electron microscopy,
                  and eventually by the success of molecular biology built on the double-helix
                  model. Any individual experiment might be wrong. But when five different
                  methods all point in the same direction, the probability that they are all
                  wrong in the same way drops sharply. Science is resilient not because each
                  brick is solid, but because the wall is built with redundancy.
                </li>
                <li className="leading-relaxed">
                  <strong>Engineering as a filter.</strong> When a scientific finding is used
                  to build something&mdash;a drug, a chip, a rocket&mdash;reality provides
                  a merciless check. The chip either works at 3 nanometers or it
                  doesn&rsquo;t. The drug either reduces mortality in a Phase III trial or it
                  doesn&rsquo;t. This means the findings that matter most are precisely the
                  ones that get tested most ruthlessly. The irreproducible results tend to
                  cluster in areas where the feedback loop is weak: social priming studies,
                  candidate gene associations, nutritional epidemiology. Where the feedback
                  loop is strong&mdash;where someone is betting real money or real lives on
                  the result&mdash;the bad findings get filtered out. Science self-corrects,
                  but it self-corrects through use, not through replication.
                </li>
                <li className="leading-relaxed">
                  <strong>Tacit knowledge and craft.</strong> A surprising amount of what
                  makes science work lives not in the published paper but in the hands and
                  intuitions of the people doing the work. A protocol that &ldquo;fails to
                  replicate&rdquo; in a different lab may work perfectly well in the original
                  lab because the original team knows a hundred small things&mdash;how to
                  handle the cells, what temperature the reagent actually needs to be, when
                  the instrument is drifting&mdash;that never made it into the methods section.
                  This is not fraud. It is the difference between a recipe and knowing how to
                  cook. It means that &ldquo;failure to replicate&rdquo; is sometimes a
                  failure of knowledge transfer, not a failure of the original result.
                </li>
                <li className="leading-relaxed">
                  <strong>Theoretical coherence.</strong> Scientists do not evaluate findings
                  in isolation. A result that is consistent with a well-established theoretical
                  framework gets more weight than one that appears out of nowhere. When a
                  new cancer drug fails to replicate in an independent trial, oncologists
                  ask whether the mechanism of action makes sense given what is known about
                  the pathway. When it does, the failure might be attributed to differences
                  in protocol or patient population. When it doesn&rsquo;t, the original
                  finding is treated with more skepticism. The theoretical web acts as a
                  prior&mdash;an implicit Bayesian filter that down-weights surprising results
                  and up-weights expected ones.
                </li>
                <li className="leading-relaxed">
                  <strong>Selection through citation and use.</strong> The scientific
                  literature is not a flat archive where every paper counts equally. In
                  practice, a small fraction of papers are cited heavily, taught in textbooks,
                  and used as foundations for further work. The rest are effectively ignored.
                  This is a form of natural selection: findings that are useful, that lead
                  to further discoveries, that enable technology, get amplified. Findings
                  that are noisy, marginal, or wrong tend to fade. The literature as a whole
                  may have a high false-positive rate, but the working knowledge that
                  scientists actually rely on is a much smaller, much more curated set.
                </li>
              </ul>
              <p className="leading-relaxed mb-4">
                The picture that emerges is not reassuring in a simple way. Science does not
                work because each individual finding is verified and true. It works because
                the system as a whole has enough redundancy, enough feedback loops, and enough
                competitive pressure that the errors wash out over time&mdash;at least in the
                domains where the stakes are high enough to force the issue. Veracity at the
                level of individual claims is less important than the resilience of the
                overall network.
              </p>
              <p className="leading-relaxed mb-4">
                This has a direct implication for the platform. If science&rsquo;s actual
                error-correction mechanism is not &ldquo;every result gets replicated&rdquo;
                but rather &ldquo;converging evidence across methods, filtered by engineering
                and theoretical coherence,&rdquo; then a conjecture market does not need to
                guarantee the veracity of any individual claim. It needs to do what science
                already does, but more efficiently: aggregate evidence from multiple sources,
                weight claims by their predictive track record, and let the price reflect the
                current state of convergence. The market is not replacing the
                error-correction mechanism. It is making it legible.
              </p>

              <h3
                id="platform-still-matters"
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
        </article>
      </div>
    </div>
  )
}
