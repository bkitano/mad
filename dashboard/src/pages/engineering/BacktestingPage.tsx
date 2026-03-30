import { Link } from 'react-router-dom'
import { InlineMath } from 'react-katex'

export default function BacktestingPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-8">
          <Link
            to="/engineering"
            className="text-sm hover:underline"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
          >
            &larr; Engineering
          </Link>
        </div>

        <header className="mb-12">
          <h1
            className="text-3xl font-bold mb-4 leading-tight"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
          >
            Backtesting
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Simulating 200 years of science to validate the market mechanism
            against historical data.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* --- The goal --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The goal
          </h2>

          <p className="leading-relaxed">
            Before deploying the market with real participants, we need to
            know whether the mechanism produces the behaviors we want. The
            only way to test this is to run it against data where we already
            know what happened &mdash; the history of science itself.
          </p>

          <p className="leading-relaxed">
            The backtesting system replays the last 200 years of scientific
            discovery through the conjecture market. Papers become evidence
            submissions. Citation relationships become dependency graphs.
            The temporal ordering of publications becomes the sequence of
            trades. The eventual consensus of the scientific community
            becomes the ground truth against which we score the market&rsquo;s
            predictions.
          </p>

          <p className="leading-relaxed">
            If the market mechanism is well-designed, it should produce three
            outcomes on historical data: (1) conjectures that the community
            eventually accepted should have rising credence over time, (2)
            participants who were &ldquo;right early&rdquo; (cited before
            the consensus formed) should have high portfolio value, and (3)
            the entropy-weighted reward function should rank contributors in
            a way that roughly tracks their actual historical impact.
          </p>

          {/* --- Data sources --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Data sources
          </h2>

          <p className="leading-relaxed">
            The simulation requires three kinds of historical data:
          </p>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              1. Conjectures
            </h3>
            <p className="leading-relaxed mb-3">
              The claims that science has debated. These are extracted from
              the historical record: key hypotheses, theories, and
              predictions that were proposed and eventually accepted or
              rejected. Sources include history-of-science literature,
              Nobel Prize citations, and major review papers that
              retrospectively identify turning points.
            </p>
            <p className="leading-relaxed">
              Each conjecture gets a creation date (when the claim was first
              proposed) and a resolution trajectory (how the community&rsquo;s
              belief evolved over time). For well-documented cases like
              continental drift, germ theory, or the Higgs boson, these
              trajectories can be reconstructed in detail. For less
              documented cases, we use coarser approximations based on
              citation patterns and textbook adoption.
            </p>
          </div>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              2. Evidence events
            </h3>
            <p className="leading-relaxed mb-3">
              Papers, experiments, and observations that moved the
              community&rsquo;s beliefs. The primary data source is
              bibliometric databases: Semantic Scholar, OpenAlex, and arXiv
              provide publication dates, citation graphs, and abstract text.
              Each paper is an evidence event that potentially updates one
              or more conjectures.
            </p>
            <p className="leading-relaxed">
              The challenge is mapping papers to conjectures. A paper on
              &ldquo;antibiotic treatment of gastric ulcers&rdquo; is
              evidence for &ldquo;H. pylori causes ulcers.&rdquo; This
              mapping can be done with a combination of citation analysis
              (papers that cite the original conjecture proposal are likely
              evidence for or against it) and semantic similarity (papers
              whose abstracts discuss the same claims).
            </p>
          </div>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              3. Dependency structure
            </h3>
            <p className="leading-relaxed mb-3">
              Which conjectures depend on which. Citation graphs are the
              primary signal: if paper B cites paper A, and both propose
              conjectures, there is likely a dependency. Co-citation analysis
              (papers that are frequently cited together) reveals implicit
              dependencies that neither paper explicitly states.
            </p>
            <p className="leading-relaxed">
              The dependency graph does not need to be perfect. The
              backtesting system can evaluate the market mechanism under
              different assumed dependency structures and measure how
              sensitive the results are to graph quality. If the mechanism
              is robust, it should produce reasonable outcomes even with a
              noisy dependency graph.
            </p>
          </div>

          {/* --- Simulation architecture --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Simulation architecture
          </h2>

          <p className="leading-relaxed">
            The simulation proceeds in chronological order. For each time
            step (e.g., one month):
          </p>

          <ol className="space-y-3 ml-6 list-decimal">
            <li className="leading-relaxed">
              <strong>Create new conjectures.</strong> Any conjectures whose
              historical proposal date falls in this time step are added to
              the market at credence 0.50 with maximum entropy.
            </li>
            <li className="leading-relaxed">
              <strong>Submit evidence.</strong> Papers published in this
              time step are submitted as evidence to the conjectures they
              are mapped to. The market updates credences based on the
              evidence.
            </li>
            <li className="leading-relaxed">
              <strong>Simulate trades.</strong> Synthetic participants
              trade based on the evidence. The simplest model: each paper&rsquo;s
              authors buy YES shares of the conjectures their paper supports
              and NO shares of the ones it contradicts. More sophisticated
              models can simulate different participant types (believers,
              skeptics, followers).
            </li>
            <li className="leading-relaxed">
              <strong>Propagate through the dependency graph.</strong>{' '}
              Evidence for conjecture B propagates to conjecture A based on
              the inferred dependency structure. Credences update upstream
              and downstream.
            </li>
            <li className="leading-relaxed">
              <strong>Record metrics.</strong> Portfolio values, credence
              trajectories, entropy curves, and impact scores are recorded
              for analysis.
            </li>
          </ol>

          {/* --- Synthetic participants --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Synthetic participants
          </h2>

          <p className="leading-relaxed">
            Real historical scientists become synthetic market participants.
            Each author in the bibliometric database gets a portfolio. Their
            trades are derived from their publications: if you published a
            paper supporting conjecture B, you are modeled as having bought
            YES shares of B (and its bundle dependencies) at the time of
            publication.
          </p>

          <p className="leading-relaxed">
            This gives us a natural test of the reward function. If the
            market is well-designed:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Barry Marshall</strong> should have high portfolio
              value &mdash; he bought YES on H. pylori at peak entropy and
              submitted evidence that moved the credence.
            </li>
            <li className="leading-relaxed">
              <strong>Alfred Wegener</strong> should have high portfolio
              value &mdash; he bought YES on continental drift decades before
              the consensus formed.
            </li>
            <li className="leading-relaxed">
              <strong>A prolific but undistinguished author</strong> who
              published many papers in well-established areas should have
              moderate portfolio value &mdash; they entered at low entropy
              (consensus already formed) with little directional reward
              potential.
            </li>
            <li className="leading-relaxed">
              <strong>An author who was consistently wrong</strong> &mdash;
              who published papers supporting conjectures that the community
              later rejected &mdash; should have low or negative portfolio
              value.
            </li>
          </ul>

          <p className="leading-relaxed">
            If the portfolio value ranking of synthetic participants roughly
            tracks the scientific community&rsquo;s retrospective assessment
            of who contributed the most, the mechanism is doing its job.
          </p>

          {/* --- What we measure --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            What we measure
          </h2>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Credence calibration.</strong> For conjectures that
              were eventually accepted (credence should approach 1) or
              rejected (credence should approach 0), does the market get
              there? How fast? Does the credence trajectory match the
              historical consensus trajectory?
            </li>
            <li className="leading-relaxed">
              <strong>Reward distribution.</strong> Does the entropy-weighted
              directional reward function produce a ranking of participants
              that makes historical sense? Are early contrarians (Marshall,
              Wegener, McClintock) rewarded more than late adopters?
            </li>
            <li className="leading-relaxed">
              <strong>Entropy dynamics.</strong> Does entropy decrease as
              evidence accumulates, as expected? Are there pathological cases
              where entropy increases or oscillates without new information?
            </li>
            <li className="leading-relaxed">
              <strong>Dependency graph quality.</strong> Does the
              bundle-inferred dependency graph match the known structure of
              scientific dependencies? Where does it diverge, and why?
            </li>
            <li className="leading-relaxed">
              <strong>Mechanism sensitivity.</strong> How sensitive are the
              results to the liquidity parameter{' '}
              <InlineMath math="b" />, the normalization function (softmax
              vs. power normalization), and the bundle discount structure?
              Which parameters produce the best historical fit?
            </li>
            <li className="leading-relaxed">
              <strong>Abuse resistance.</strong> If we inject synthetic
              manipulators (participants who try to move credences without
              evidence), how robust is the mechanism? Does the entropy cost
              make manipulation expensive enough?
            </li>
          </ul>

          {/* --- Scope and limitations --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Scope and limitations
          </h2>

          <p className="leading-relaxed">
            The backtesting system is a simulation, not a replay. Real
            market participants would behave differently from synthetic
            participants whose trades are derived from publication records.
            The simulation cannot capture strategic behavior, insider
            knowledge, or the social dynamics of scientific communities.
          </p>

          <p className="leading-relaxed">
            The data has survivorship bias: we know which conjectures
            eventually won, which makes it easy to construct a dataset
            where the market &ldquo;works.&rdquo; The harder test is
            whether the mechanism produces good behavior for conjectures
            that are still open &mdash; where we don&rsquo;t know the
            answer. The backtesting system can partially address this by
            truncating the simulation at various historical dates and
            measuring the market&rsquo;s predictions against what happened
            next, but this is still retrospective.
          </p>

          <p className="leading-relaxed">
            Despite these limitations, the backtest is the best available
            tool for tuning mechanism parameters before live deployment. If
            the mechanism fails on historical data &mdash; if it rewards
            the wrong people, produces miscalibrated credences, or is
            easily manipulated &mdash; it will certainly fail in production.
            Passing the backtest is necessary, though not sufficient.
          </p>

          {/* --- Implementation phases --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Implementation phases
          </h2>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Phase 1: Single-domain proof of concept
            </h3>
            <p className="leading-relaxed">
              Pick one well-documented scientific controversy (e.g., H.
              pylori and ulcers, 1982&ndash;2005) and simulate it end to
              end. This validates the pipeline: conjecture creation, evidence
              mapping, trade simulation, credence updates, and portfolio
              scoring. The domain is small enough to verify every step by
              hand.
            </p>
          </div>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Phase 2: Multi-domain historical simulation
            </h3>
            <p className="leading-relaxed">
              Expand to 10&ndash;20 major scientific developments across
              physics, biology, medicine, and computer science. This tests
              the mechanism at scale: cross-domain dependencies, overlapping
              conjectures, and the interaction between many simultaneous
              participants. The dependency graph becomes non-trivial and
              the bundle discount mechanism gets a real workout.
            </p>
          </div>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Phase 3: Full bibliometric backtest
            </h3>
            <p className="leading-relaxed">
              Run the market against the full bibliometric record: millions
              of papers, hundreds of thousands of authors, thousands of
              implicit conjectures. This is the stress test. Most conjectures
              will need to be extracted automatically (from paper titles,
              abstracts, and claims), and the evidence mapping will be
              entirely algorithmic. The goal is not perfect accuracy but
              statistical validation: does the mechanism produce sensible
              distributions of credence, reward, and impact at scale?
            </p>
          </div>

        </div>
      </div>
    </div>
  )
}
