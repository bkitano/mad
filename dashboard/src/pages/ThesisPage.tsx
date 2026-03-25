import { useState, useEffect } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

interface Section {
  id: string
  title: string
  level: number
}

const sections: Section[] = [
  { id: 'overview', title: 'Overview', level: 1 },
  { id: 'challenges-today', title: 'Challenges and Limitations in Science Today', level: 1 },
  { id: 'reproduceability', title: 'Reproduceability', level: 2 },
  { id: 'peer-review-delay', title: 'Peer Review Delay', level: 2 },
  { id: 'high-barriers', title: 'High Barriers to Entry', level: 2 },
  { id: 'null-results', title: 'Lack of Null Result Publications', level: 2 },
  { id: 'non-perfect-confidence', title: 'Lack of Non-Perfect Confidence Publications', level: 2 },
  { id: 'challenges-future', title: 'Challenges and Opportunities in the Near Future', level: 1 },
  { id: 'peer-review-collapse', title: 'Peer Review Collapse', level: 2 },
  { id: 'ai-faster-signals', title: 'AI Needs Faster Signals', level: 2 },
  { id: 'designing-markets', title: 'Designing Markets via Microeconomics', level: 1 },
  { id: 'marketplace-of-ideas', title: 'Science as a Marketplace of Ideas', level: 1 },
  { id: 'conjecture-markets', title: 'Conjecture/Belief Markets', level: 1 },
  { id: 'how-they-work', title: 'How They Work', level: 2 },
  { id: 'vs-prediction-markets', title: 'vs. Prediction Markets', level: 2 },
  { id: 'vs-stocks', title: 'vs. Stocks', level: 2 },
  { id: 'creating-value', title: 'Creating Value in Conjecture Markets', level: 1 },
  { id: 'cold-start', title: 'Solving the Cold-Start Problem', level: 1 },
  { id: 'portfolio-rl', title: 'Portfolio Value as RL Signal', level: 1 },
  { id: 'onboarding', title: 'Onboarding Real Actors', level: 1 },
]

export default function ThesisPage() {
  const [activeSection, setActiveSection] = useState('overview')

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
    <div className="thesis-page min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-6xl mx-auto px-6 py-12 flex gap-12">
        {/* Side Table of Contents */}
        <nav className="hidden lg:block w-64 flex-shrink-0">
          <div className="sticky top-24">
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
          <header className="mb-16">
            <h1
              className="text-4xl font-bold mb-4 leading-tight"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
            >
              Science 2028
            </h1>
            <p
              className="text-lg leading-relaxed"
              style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
            >
              A position paper on scientific discovery, veracity, and market mechanisms in the age of AI.
            </p>
            <div
              className="mt-4 text-sm"
              style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}
            >
              Last updated: March 2026 &middot; Working draft
            </div>
          </header>

          {/* 1. Overview */}
          <section id="overview" className="thesis-section mb-16">
            <h2 className="thesis-h2">1. Overview</h2>
            <p className="thesis-p">
              The promise of AI in science is great. AI agents are beginning to write proofs, run experiments,
              and generate hypotheses at a pace that no individual researcher can match. But the institutions
              of science&mdash;peer review, publication, citation, reputation&mdash;were designed for a world
              where humans were both the producers and the consumers of knowledge. As AI accelerates the
              production side, the bottleneck shifts to evaluation, verification, and consensus.
            </p>
            <p className="thesis-p">
              We are pre-paradigmatic on AI science research. There is no steam engine yet. This paper
              proposes a framework for formalizing the contours of what is needed to do science in an era
              where humans are the bottleneck.
            </p>
          </section>

          {/* 2. Challenges and Limitations in Science Today */}
          <section id="challenges-today" className="thesis-section mb-16">
            <h2 className="thesis-h2">2. Challenges and Limitations in Science Today</h2>
            <p className="thesis-p">
              Much of what makes science work has to do with the fact that science is participatory:
              research benefits from having people review it, challenge its components, and reproduce it.
              Several structural problems undermine this participatory process today.
            </p>
          </section>

          <section id="reproduceability" className="thesis-section mb-12">
            <h3 className="thesis-h3">Reproduceability</h3>
            <p className="thesis-p">
              It is not obvious how individuals should evaluate or verify research results. The
              reproducibility crisis reflects a deeper problem: veracity challenges pervade modern
              science, and there are few quantifiable signals for whether a result is true. How do
              people actually know that research is true? How do you find mistakes?
            </p>
            <p className="thesis-p">
              Research and code may seem different, but they share a fundamental reliance on
              empiricism&mdash;tests, artifacts, and reproducible observations. Both ultimately depend
              on modes of verification:
            </p>
            <ul className="thesis-list">
              <li>
                <strong>Empiricism</strong> (observation of evidence that something is true)&mdash;&ldquo;if
                you can&rsquo;t tie a claim to an observation or experience, don&rsquo;t trust it&rdquo;
                (Hume). Strengths: anchors belief, enables falsification, scales with instrumentation.
                Limitations: induction and failure to generalize, overfitting, data without theory is limited.
              </li>
              <li>
                <strong>Theory</strong> (structure, compression, prediction)&mdash;&ldquo;internal models
                to organize and interpret observations&rdquo; (Descartes). Theory is a compression or
                topology of data, a predictive engine, and a causal story. Strengths: generalization,
                counterfactuals, guides what data to collect. Weaknesses: detached abstraction,
                underdetermination (multiple theories can explain the same thing), and motivated reasoning.
              </li>
            </ul>
          </section>

          <section id="peer-review-delay" className="thesis-section mb-12">
            <h3 className="thesis-h3">Peer Review Delay</h3>
            <p className="thesis-p">
              Today, scientists rely on paper citation count, scientist reputation, peer review, and
              reproductions as heuristics for result veracity. These are slow signals to generate: they
              are bounded by scientists reviewing results and producing new papers that cite that result,
              which itself is bottlenecked by the time to produce results and the number of scientists
              working in a given domain. It takes months to have even modest results, draft pre-prints,
              go through peer review, publish at conferences, and have your work cited to know that it
              was impactful.
            </p>
          </section>

          <section id="high-barriers" className="thesis-section mb-12">
            <h3 className="thesis-h3">High Barriers to Entry</h3>
            <p className="thesis-p">
              Access to scientific contribution has historically been gated by institutional affiliation,
              funding, equipment, and credentials. These barriers limit the number of people who can
              contribute to science. AI has the potential to dramatically lower these barriers, but only
              if the surrounding institutions can adapt.
            </p>
          </section>

          <section id="null-results" className="thesis-section mb-12">
            <h3 className="thesis-h3">Lack of Null Result Publications</h3>
            <p className="thesis-p">
              Because reviewers are limited, rarely do null findings get published and disseminated. This
              creates a systematic bias in the scientific record: the published literature overrepresents
              positive results and underrepresents the negative results that are equally informative for
              directing future work. A system that only rewards &ldquo;wins&rdquo; systematically
              misallocates resources.
            </p>
          </section>

          <section id="non-perfect-confidence" className="thesis-section mb-12">
            <h3 className="thesis-h3">Lack of Non-Perfect Confidence Publications</h3>
            <p className="thesis-p">
              Most publications don&rsquo;t come with confidence intervals, but even results with low
              confidence can still be useful and incorporated into downstream work. If confidence in the
              veracity of a result has superlinear costs (which seems true), then science conducted
              only at high confidence thresholds will be expensive. There is a difference between
              &ldquo;answers,&rdquo; &ldquo;improvements,&rdquo; and other kinds of research
              results&mdash;and the current publication system collapses these distinctions.
            </p>
          </section>

          {/* 3. Challenges, Limitations and Opportunities in the Near Future */}
          <section id="challenges-future" className="thesis-section mb-16">
            <h2 className="thesis-h2">3. Challenges and Opportunities in the Near Future</h2>
            <p className="thesis-p">
              If we experience a massive growth in the number of scientists (due to lower barriers to
              entry by AI) and the speed of research production (due to increased throughput by AI), it
              will lead to real changes for science and scientists that existing institutions
              can&rsquo;t support.
            </p>
          </section>

          <section id="peer-review-collapse" className="thesis-section mb-12">
            <h3 className="thesis-h3">Peer Review Collapse</h3>
            <p className="thesis-p">
              A 100x increase in scientific throughput without commensurate gains in review capacity
              would lead to bottlenecks on publication rate. Peer review is already collapsing under
              current AI-assisted throughput: as humans generate and publish more research, peer
              review can&rsquo;t keep up, and science stalls. The current system assumes that human
              reviewers can scale with production&mdash;an assumption that AI invalidates.
            </p>
            <p className="thesis-p">
              Agents also have technological limitations that compound this problem. They lack continual
              learning: after an AI writes a proof, if you spin up a new session, it doesn&rsquo;t
              remember how it solved the problem before. Agents either succeed or fail, but don&rsquo;t
              yet create useful partial stages. AIs excel at breadth, and humans excel at depth.
            </p>
          </section>

          <section id="ai-faster-signals" className="thesis-section mb-12">
            <h3 className="thesis-h3">AI Needs Faster Signals</h3>
            <p className="thesis-p">
              AI needs faster signals for what constitutes good research. It takes months to have even
              modest results, draft pre-prints, go through peer review, publish at conferences, and have
              your work cited to know that it was impactful. For AI agents to improve at doing science,
              they need tighter feedback loops than the current publication cycle provides.
            </p>
            <p className="thesis-p">
              How should research agents do work that is well-structured and aligned? Key considerations
              include pedagogical viability (how well do results lend themselves to being interpreted by
              humans?), minimizing consensus-building friction (how can you create artifacts that minimize
              the effort required for humans to accept their veracity?), and whether there is a difference
              in how humans verify versus how agents verify.
            </p>
            <div className="thesis-callout">
              <strong>Bayesian loop:</strong> theory &rarr; prediction &rarr; experiment &rarr; data
              &rarr; update theory &rarr; &hellip; The criteria for each cycle include empirical adequacy,
              falsifiability (Popper), predictive power, parsimony (Occam&rsquo;s Razor), coherence with
              other well-supported theories, and causal robustness (surviving intervention, not just
              correlation).
            </div>
          </section>

          {/* 4. Designing Markets via Microeconomics */}
          <section id="designing-markets" className="thesis-section mb-16">
            <h2 className="thesis-h2">4. Designing Markets via Microeconomics</h2>
            <p className="thesis-p">
              Microeconomics offers a mature set of principles for diagnosing why markets fail and how
              to fix them. Before applying these to science specifically, it is worth laying out the
              core concepts, because the problems described in the previous sections map onto them
              with surprising precision.
            </p>
            <p className="thesis-p">
              Foundational principles of market design hold that well-functioning markets should
              be <strong>thick</strong>, <strong>deep</strong>, <strong>uncongested</strong>,
              and <strong>safe</strong>.
            </p>
            <ul className="thesis-list">
              <li>
                <strong>Thick:</strong> enough participants and opportunities show up in one place to
                make good matches possible. A thin market has too few participants to generate useful
                signal. In science, many subfields are thin: only a handful of researchers work on a
                given problem, so results go unreviewed, unreplicated, and unchallenged for years.
                High barriers to entry (section 2) directly reduce thickness.
              </li>
              <li>
                <strong>Deep:</strong> the market has enough resting interest near the current price
                so that large orders can trade with limited price impact. A deep market absorbs new
                information smoothly; a shallow one overreacts to noise. In science, depth corresponds
                to the capacity of a field to evaluate new results without being overwhelmed. A field
                where a single flashy paper can redirect an entire community&rsquo;s attention&mdash;before
                anyone has reproduced the result&mdash;is shallow. The reproducibility crisis (section 2)
                is in part a depth problem.
              </li>
              <li>
                <strong>Uncongested:</strong> the system gives participants enough structure, time, and
                mechanism to evaluate options instead of getting overwhelmed or forced into premature
                decisions. Congestion occurs when the rate of incoming information exceeds the
                community&rsquo;s capacity to process it. This is exactly the peer review collapse
                described in section 3: a 100x increase in throughput without commensurate gains in
                review capacity creates congestion. Participants cannot evaluate what is in front of
                them, so they either ignore it or accept it uncritically.
              </li>
              <li>
                <strong>Safe:</strong> participants can act on their true preferences without excessive
                gaming, adverse selection, or fear that honest participation will punish them. A safe
                market is one where honest behavior is incentive-compatible. In science, the absence
                of safety manifests as publication bias: researchers are punished for publishing null
                results or low-confidence findings (section 2), so they withhold them. The market
                never sees the information, and everyone is worse off.
              </li>
            </ul>
            <p className="thesis-p">
              Beyond these structural properties, several core microeconomic concepts diagnose specific
              failures in how science currently operates.
            </p>
            <p className="thesis-p">
              <strong>Price discovery.</strong> In markets, price discovery occurs when transactions
              between buyers and sellers are broadcasted, and the price adjusts as new information
              about the underlying value of an asset becomes available. Determining the veracity of a
              scientific result is the same kind of discovery procedure under uncertainty: a distributed
              consensus process where many participants contribute partial information. Today, the
              &ldquo;price&rdquo; of a scientific result&mdash;the community&rsquo;s credence in
              it&mdash;is discovered through citation count, reputation, peer review, and
              reproductions. All of these are slow, bounded by human bandwidth, and provide only
              coarse-grained signal.
            </p>
            <p className="thesis-p">
              <strong>Liquidity.</strong> A liquid market lets information move into prices quickly
              because participation is easy. An epistemically liquid field lets claims get evaluated
              quickly because tools, benchmarks, data, and reviewer bandwidth are available. Illiquid
              knowledge systems have lots of claims but few decisive tests. The peer review delays
              described in section 2 are a liquidity problem: information exists (a result has been
              produced) but cannot be efficiently incorporated into the community&rsquo;s beliefs
              because the mechanisms for evaluation are too slow and too scarce.
            </p>
            <p className="thesis-p">
              <strong>Externalities.</strong> An externality occurs when a transaction affects parties
              not involved in it. Science is dominated by positive externalities: a foundational result
              creates value for everyone who builds on it, but the original author captures only a
              fraction of that value (via citations, which carry no direct compensation). This leads to
              systematic underproduction of the most broadly useful work&mdash;infrastructure, tooling,
              negative results, and careful replications&mdash;because the incentives favor narrow,
              publishable novelty.
            </p>
            <p className="thesis-p">
              <strong>Information asymmetry.</strong> Adverse selection arises when one party knows more
              about the quality of a good than the other. In science, the author of a result knows far
              more about its robustness&mdash;which experiments failed, which parameters were
              tuned, which results were cherry-picked&mdash;than any reader does. Peer review is the
              primary mechanism for closing this gap, but it is slow, noisy, and does not scale. A
              well-designed system would make verification cheaper so that the asymmetry narrows faster.
            </p>
            <p className="thesis-p">
              Each of the problems identified in sections 2 and 3&mdash;reproducibility, peer review
              delay, barriers to entry, publication bias, review collapse, slow feedback&mdash;maps
              onto a recognized market failure. This is not coincidence: science is a system of
              distributed agents exchanging information under uncertainty, which is precisely what
              markets are. The question is whether we can design the market better.
            </p>
          </section>

          {/* 5. Science as a Marketplace of Ideas */}
          <section id="marketplace-of-ideas" className="thesis-section mb-16">
            <h2 className="thesis-h2">5. Science as a Marketplace of Ideas</h2>
            <p className="thesis-p">
              With this microeconomic vocabulary in hand, we can now make the analogy explicit.
              Define a <strong>scientist</strong> as a participant active in creating new results and
              adding them to the marketplace of ideas, and <strong>science</strong> as the marketplace
              itself. Scientists participate in science by exchanging, evaluating, and incorporating
              results of others. Some common metrics to measure scientists are productivity (results
              per time unit), citation count, and h-index. Some metrics to measure science (the
              marketplace) include the number of scientists, the result adoption velocity (how quickly
              results are integrated by other scientists), and the idea exchange network connectivity
              (how much scientists exchange with each other).
            </p>
            <p className="thesis-p">
              The mapping from the previous section is direct. The <strong>price</strong> of an asset
              corresponds to the credence a community places in a result&mdash;not a perfect measure,
              but an emergent consensus that aggregates the beliefs of many participants. Today, this
              credence is approximated crudely by citation count, journal prestige, and author
              reputation. A <strong>transaction</strong> corresponds to one researcher incorporating
              another&rsquo;s result into their own work&mdash;citing it, building on it, reproducing
              it. Each such exchange is an implicit endorsement, a signal that the result was valuable
              enough to use.
            </p>
            <p className="thesis-p">
              The question this framing poses is: can we construct a mechanism that aggregates
              distributed belief about the veracity of results more efficiently than these proxies
              do? Can we make the marketplace thicker, deeper, less congested, and safer&mdash;so
              that it supports both a rapid increase in participants and a rapid acceleration of
              individual production, without sacrificing the quality of science as a whole?
            </p>
          </section>

          {/* 6. Conjecture/Belief Markets */}
          <section id="conjecture-markets" className="thesis-section mb-16">
            <h2 className="thesis-h2">6. Conjecture/Belief Markets</h2>
            <p className="thesis-p">
              Scientific claims rarely admit final resolution; they remain revisable in light of new
              evidence. Therefore a conjecture market should not rely on binary settlement. Instead, it
              should reward contributors for marginal improvements in the predictive performance and
              calibration of the conjecture network over time. Compensation is thus provisional and
              continuously updated, reflecting whether a contribution is borne out by later evidence,
              descendant conjectures, and empirical predictions.
            </p>
          </section>

          <section id="how-they-work" className="thesis-section mb-12">
            <h3 className="thesis-h3">How They Work</h3>
            <p className="thesis-p">
              All research aims to invalidate hypotheses about the world. A conjecture market
              rephrases scientific claims into conjectures with predictive capacity&mdash;falsifiable
              statements that can be traded. Each conjecture has a live price reflecting the
              community&rsquo;s current credence in it.
            </p>
            <p className="thesis-p">
              The core mechanism is simple: you buy exposure to ideas you depend on before you
              publish. If a scientist has a new result that supports an existing conjecture, they
              buy that conjecture before publishing their work. If the work is good, the price
              moves, and they benefit from the delta. If participants act rationally, prices will
              reflect the community&rsquo;s best estimate of each conjecture&rsquo;s veracity,
              and portfolios will reflect each participant&rsquo;s scientific judgment.
            </p>
          </section>

          <section id="vs-prediction-markets" className="thesis-section mb-12">
            <h3 className="thesis-h3">What Makes Them Different from Prediction Markets?</h3>
            <p className="thesis-p">
              Conjecture markets don&rsquo;t resolve. Prediction markets (event contracts) settle on a
              binary outcome: an event either happens or it doesn&rsquo;t. Scientific conjectures
              rarely admit final resolution&mdash;they remain revisable in light of new evidence. The
              price of a conjecture is never &ldquo;settled&rdquo;; it is always the
              community&rsquo;s current best estimate, subject to change as new work is published.
            </p>
          </section>

          <section id="vs-stocks" className="thesis-section mb-12">
            <h3 className="thesis-h3">What Makes Them Different from Stocks?</h3>
            <p className="thesis-p">
              Conjecture markets don&rsquo;t pay dividends. A stock represents ownership of a
              cash-flow-generating entity. A conjecture represents a claim about the world. Its value
              comes entirely from the community&rsquo;s evolving credence in it&mdash;there is no
              underlying revenue stream, only the changing beliefs of participants as new evidence
              arrives.
            </p>
          </section>

          {/* 7. Creating Value in Conjecture Markets */}
          <section id="creating-value" className="thesis-section mb-16">
            <h2 className="thesis-h2">7. Creating Value in Conjecture Markets</h2>
            <p className="thesis-p">
              Publishing evidence that yields a change in conjecture prices is impactful research. The
              process is simple: you buy exposure to the conjectures your work depends on, publish your
              research, and benefit from the resulting price movement. If the work is good, the
              conjectures it supports go up; if it undermines them, they go down. Either way, the
              scientist who did the work and positioned accordingly captures the value.
            </p>
            <p className="thesis-p">
              Consider this from a Bayesian perspective. Let <InlineMath math="A" /> be a conjecture
              currently priced at <InlineMath math="P(A) = 0.8" />. A scientist produces a new
              result <InlineMath math="B" /> that depends on <InlineMath math="A" /> alongside other
              conjectures <InlineMath math="X" />, and through independent work believes{' '}
              <InlineMath math="B" /> is true at 80%. The posterior is:
            </p>
            <div className="thesis-equation">
              <BlockMath math="P(A \mid B, X) = \frac{P(B \mid A, X)\,P(A \mid X)}{P(B \mid X)}" />
            </div>
            <p className="thesis-p">
              Expanding the denominator:
            </p>
            <div className="thesis-equation">
              <BlockMath math="P(B \mid X) = P(B \mid A, X)\,P(A \mid X) + P(B \mid \neg A, X)\,P(\neg A \mid X)" />
            </div>
            <p className="thesis-p">
              So:
            </p>
            <div className="thesis-equation">
              <BlockMath math="P(A \mid B, X) = \frac{0.64}{0.64 + 0.2\,P(B \mid \neg A, X)}" />
            </div>
            <p className="thesis-p">
              If <InlineMath math="B" /> is likely to be true regardless of
              whether <InlineMath math="A" /> is true, the posterior on <InlineMath math="A" /> doesn&rsquo;t
              change. But if <InlineMath math="B" /> heavily depends on <InlineMath math="A" /> being
              true, and we observe <InlineMath math="B" />, then <InlineMath math="A" /> should be
              upweighted. The key insight is that scientists never personally report these
              probabilities&mdash;they are downstream of the prices of the conjectures they trade.
              If we can observe which conjectures researchers buy before they publish, this gives us
              an implied posterior: a revealed belief, expressed through action rather than self-report.
              Over time, these purchasing patterns form a dependency graph of beliefs across the
              entire scientific community.
            </p>
            <p className="thesis-p">
              Having a portfolio of conjectures that increases in value is equivalent to having
              scientific intuition. A scientist whose portfolio consistently appreciates has
              demonstrated an ability to identify which conjectures will be borne out by future
              evidence&mdash;the hallmark of good scientific judgment. Attribution, credit, and
              compensation do not need to be engineered into the mechanism: if participants act
              rationally, prices and portfolios represent all of that.
            </p>
          </section>

          {/* 8. Solving the Cold-Start Problem */}
          <section id="cold-start" className="thesis-section mb-16">
            <h2 className="thesis-h2">8. Solving the Cold-Start Problem</h2>
            <p className="thesis-p">
              The cold-start problem: how do you get participants into your market if the market
              doesn&rsquo;t exist?
            </p>
            <p className="thesis-p">
              The answer is simulation backtesting on real historical participants&rsquo; implied
              trading behavior. If we can reconstruct what scientists historically would have traded
              based on their publication records, citation patterns, and research trajectories, we can
              bootstrap a synthetic market with realistic dynamics.
            </p>
            <p className="thesis-p">
              Building a high-fidelity simulation means we can derive explicit alpha scores from
              research and portfolio values for scientists from their implied behavior. The simulation
              provides the initial liquidity and price discovery that a live market would otherwise
              lack, giving new participants a meaningful market to enter from day one.
            </p>
          </section>

          {/* 9. Portfolio Value as RL Signal */}
          <section id="portfolio-rl" className="thesis-section mb-16">
            <h2 className="thesis-h2">9. Portfolio Value as RL Signal for Training Better AI Scientists</h2>
            <p className="thesis-p">
              With a high-fidelity simulation, we can assemble time series datasets and train agents
              to simulate trades in conjectures. This is how agents can learn an intuition for what
              fields will be fruitful.
            </p>
            <p className="thesis-p">
              If we determine a way to estimate the cost of research, then we can train agents
              to maximize conjecture-alpha per unit cost. This is how agents can learn to decide which
              experiments to pursue:
            </p>
            <div className="thesis-equation">
              <BlockMath math="\max_{\pi} \; \mathbb{E}\!\left[\sum_t \gamma^t \frac{\Delta \text{Portfolio}_t}{\text{Cost}_t}\right]" />
            </div>
            <p className="thesis-p">
              where <InlineMath math="\pi" /> is the agent&rsquo;s policy for selecting which conjectures
              to investigate and what evidence to produce,{' '}
              <InlineMath math="\Delta \text{Portfolio}_t" /> is the change in portfolio value from the
              agent&rsquo;s actions, and <InlineMath math="\text{Cost}_t" /> is the estimated cost of
              the research conducted. The simulation provides the training environment, and portfolio
              value provides the reward signal.
            </p>
          </section>

          {/* 10. Onboarding Real Actors */}
          <section id="onboarding" className="thesis-section mb-16">
            <h2 className="thesis-h2">10. Onboarding Real Actors into the Live Market</h2>
            <p className="thesis-p">
              With a high-fidelity simulation on historical data, we can start to onboard present
              actors with novel conjectures. The transition from simulation to live market requires
              careful consideration of different stakeholder groups:
            </p>
            <ul className="thesis-list">
              <li>
                <strong>For academics:</strong> the market provides faster feedback signals than the
                traditional publication cycle, continuous credit attribution through ownership shares,
                and a mechanism for even partial or negative results to generate value.
              </li>
              <li>
                <strong>For individual contributors:</strong> lower barriers to entry than traditional
                academic publishing, the ability to contribute evidence and earn ownership without
                institutional affiliation, and transparent, market-based evaluation of contributions.
              </li>
              <li>
                <strong>For agents:</strong> a well-defined reward signal (portfolio value) that can
                be optimized through reinforcement learning, a structured environment for producing
                and evaluating research artifacts, and integration into the broader scientific
                ecosystem through the dependency graph.
              </li>
            </ul>
          </section>

          <footer className="mt-20 pt-8 border-t" style={{ borderColor: 'var(--paper-deep)' }}>
            <p className="text-sm" style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}>
              These notes are a living document, updated as new experiments complete and new
              insights emerge. For the latest experimental results, see the{' '}
              <a href="/" className="underline" style={{ color: 'var(--accent)' }}>
                platform dashboard
              </a>
              .
            </p>
          </footer>
        </article>
      </div>
    </div>
  )
}
