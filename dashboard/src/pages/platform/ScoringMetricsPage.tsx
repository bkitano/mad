import { Link } from 'react-router-dom'
import { BlockMath, InlineMath } from 'react-katex'

export default function ScoringMetricsPage() {
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
            Scoring &amp; Metrics
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            How the conjecture market measures performance: portfolio value,
            attribution graphs, trade history, and conjecture impact.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* --- Credence, Cost, Reward --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Three numbers, not one
          </h2>

          <p className="leading-relaxed">
            The scoring system rests on the separation described in{' '}
            <Link
              to="/platform/market-incentives"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Market Incentives
            </Link>
            : each conjecture carries a <strong>credence</strong> (the
            community&rsquo;s aggregate belief, between 0 and 1), each
            position has an entropy-derived <strong>cost</strong> (what you
            paid to enter), and each participant earns a directional{' '}
            <strong>reward</strong> (how much the consensus moved toward
            their position after entry). These are not the same number.
            The scoring system is built on the reward, not the credence.
          </p>

          {/* --- Portfolio Value --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Portfolio Value
          </h2>

          <p className="leading-relaxed">
            Your portfolio value measures how well your scientific judgment
            has tracked with the evolving evidence. It is <em>not</em> simply
            shares times current credence &mdash; that would reward anyone
            who accumulates cheap consensus positions. Instead, portfolio
            value is a function of <em>when</em> you entered each position,{' '}
            <em>how much uncertainty you bore</em>, and <em>how far the
            consensus moved toward you</em>.
          </p>

          <p className="leading-relaxed">
            For each position, the contribution to portfolio value depends on
            three quantities: the entropy at entry{' '}
            <InlineMath math="H(P_{t_0})" />, the credence at entry{' '}
            <InlineMath math="P_{t_0}(A)" />, and the current credence{' '}
            <InlineMath math="P_t(A)" />. The directional reward for a YES
            position is:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="r_i(A, t) = H(P_{t_0}) \cdot \big(P_t(A) - P_{t_0}(A)\big)" />
          </div>

          <p className="leading-relaxed">
            The entropy term <InlineMath math="H(P_{t_0})" /> weights by
            how much uncertainty you bought into. The directional
            term <InlineMath math="P_t(A) - P_{t_0}(A)" /> captures whether
            the consensus moved toward your position (positive) or away from
            it (negative). A NO position uses{' '}
            <InlineMath math="P_{t_0}(A) - P_t(A)" /> instead. Total
            portfolio value is:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="V_t = \sum_{A \in \text{Portfolio}} n_i(A) \cdot r_i(A, t)" />
          </div>

          <p className="leading-relaxed">
            This means two participants holding the same number of shares of
            the same conjecture at the same current credence can have very
            different portfolio values, because they entered at different
            times under different levels of uncertainty. The participant who
            bought at <InlineMath math="H = 1.0" /> and rode the credence
            from 0.50 to 0.90 has a far more valuable position than one who
            bought at <InlineMath math="H = 0.08" /> when the credence was
            already 0.95.
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
              You hold 50 YES shares of &ldquo;CRISPR base editing achieves
              &gt;90% efficiency in primary human T cells&rdquo; bought when
              credence was 0.30 (<InlineMath math="H \approx 0.88" /> bits).
              Current credence is 0.55. You also hold 30 YES shares of
              &ldquo;Long-context transformers scale sub-quadratically&rdquo;
              bought when credence was 0.60 (<InlineMath math="H \approx 0.97" />{' '}
              bits). Current credence is 0.65.
            </p>
            <p className="leading-relaxed">
              Your portfolio value is{' '}
              <InlineMath math="50 \times 0.88 \times (0.55 - 0.30) + 30 \times 0.97 \times (0.65 - 0.60) = 11.0 + 1.46 = 12.46" />.
              The CRISPR position dominates because you entered at higher
              entropy and the consensus has moved further toward you. The
              transformer position is worth less despite high entry entropy,
              because the credence has only moved 0.05 in your direction. If
              new results push the CRISPR conjecture to 0.70, that position
              alone rises to{' '}
              <InlineMath math="50 \times 0.88 \times 0.40 = 17.6" />{' '}
              &mdash; a gain driven by evidence confirming your early conviction.
            </p>
          </div>

          {/* --- Trade History --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Trade History
          </h2>

          <p className="leading-relaxed">
            Every position change is recorded as a trade. Your trade history
            is the complete log of when you bought or sold positions, the
            entropy at the time of entry, the direction of your position, and
            what evidence you attached. This record serves several purposes:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Calibration tracking.</strong> Over many trades, the
              market can assess whether you are well-calibrated &mdash; do
              your directional bets track with where the consensus eventually
              moves? Long-run calibration determines your trust weight in
              the system.
            </li>
            <li className="leading-relaxed">
              <strong>Entropy at entry.</strong> The timestamp and the
              credence distribution at the time of each trade determine the
              entropy you bought into. This is the core input to the reward
              function and cannot be reconstructed without the trade record.
            </li>
          </ul>

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
              In 1982, Barry Marshall and Robin Warren proposed that stomach
              ulcers were caused by the bacterium <em>Helicobacter
              pylori</em>, not by stress or diet. Suppose the conjecture had
              been on the market.
            </p>
            <p className="leading-relaxed mb-4">
              At proposal, no one owns shares. Credence:{' '}
              <InlineMath math="P = 0.50" />. Entropy:{' '}
              <InlineMath math="H = 1.0" /> bit. Marshall buys YES at peak
              uncertainty. His entry cost is high &mdash; he is buying into
              maximum entropy. Simultaneously, establishment
              gastroenterologists buy NO, equally convinced. Both sides pay
              the same entropy-derived cost.
            </p>
            <p className="leading-relaxed mb-4">
              Marshall submits evidence: he cultures the bacterium, drinks a
              petri dish of it, develops gastritis. The credence shifts, but
              slowly &mdash; most of the community doubles down on the
              stress hypothesis. The NO holders are paying into real
              uncertainty, but their position is directionally wrong. As
              RCTs accumulate through the 1990s and credence climbs toward
              0.90, Marshall&rsquo;s reward grows:{' '}
              <InlineMath math="H(P_{t_0}) \cdot (P_t - P_{t_0}) = 1.0 \times (0.90 - 0.50) = 0.40" />{' '}
              per share. The NO holders&rsquo; positions lose value by
              the same logic: they bore entropy in the wrong direction.
            </p>
            <p className="leading-relaxed">
              His trade history shows conviction at peak entropy, evidence
              submission that moved the credence, and directional reward
              that accumulated as the field caught up. The establishment
              doctors who bought NO at the same entropy have the opposite
              record &mdash; same cost, opposite return.
            </p>
          </div>

          {/* --- Total Conjecture Impact --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Total Conjecture Impact
          </h2>

          <p className="leading-relaxed">
            Impact is bidirectional. It flows <em>upstream</em>: when your
            evidence for conjecture B improves the posterior on conjecture A
            (because B depends on A), that improvement to A counts as your
            impact. And it flows <em>downstream</em>: when future
            conjectures are created that depend on your conjecture A, and
            evidence on those downstream conjectures generates trading
            activity and credence updates, the value that propagates back up
            to A is also your impact. You earn impact in both directions
            &mdash; for the conjectures you build on, and for the
            conjectures that build on you.
          </p>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The counterfactual
          </h3>

          <p className="leading-relaxed">
            The core idea is simple: we compare the world where you
            contributed to the world where you didn&rsquo;t. The hard part
            is constructing that second world. Let&rsquo;s walk through it
            step by step.
          </p>

          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Example: CRISPR efficiency
            </h3>

            <p className="leading-relaxed mb-4">
              Consider the conjecture B: &ldquo;CRISPR base editing achieves
              &gt;90% efficiency in primary human T cells.&rdquo; It depends
              on conjecture A: &ldquo;Cas9 variants can be engineered for
              higher fidelity.&rdquo; B has one key observable
              implication <InlineMath math="y" />: the next published
              efficiency benchmark for base editing in T cells.
            </p>

            <p className="leading-relaxed mb-4">
              <strong>Step 1: Snapshot the market before you act.</strong>{' '}
              The credence on B is <InlineMath math="P = 0.30" /> (entropy{' '}
              <InlineMath math="H \approx 0.88" /> bits). The credence on A
              is <InlineMath math="P = 0.50" />. This is the state of the
              world without your contribution. We record this as the
              counterfactual forecast:{' '}
              <InlineMath math="p^{(-i)}_t(y) = 0.30" /> &mdash; in the
              world without you, the market thinks there is a 30% chance the
              next benchmark exceeds 90%.
            </p>

            <p className="leading-relaxed mb-4">
              <strong>Step 2: You contribute.</strong> You submit a paper
              showing a new Cas9 variant with dramatically higher fidelity.
              The market reads the evidence and updates B to{' '}
              <InlineMath math="P(B) = 0.55" />. The gap between 0.55
              and 0.30 is your claimed marginal improvement. But a claim is
              not impact &mdash; the question is whether the community
              eventually converges toward your update or away from it.
            </p>

            <p className="leading-relaxed mb-4">
              <strong>Step 3: Downstream evidence validates you.</strong>{' '}
              Six months later, an independent lab creates conjecture D:
              &ldquo;Lab X achieved 93% base editing efficiency in primary
              human T cells using protocol Y.&rdquo; This is a specific,
              verifiable claim. If the community finds D credible &mdash;
              the protocol is reproducible, the data is clean &mdash; then
              D trivially updates the posterior on B. Not causally (D
              didn&rsquo;t make B true), but syllogistically: if D is true,
              then B is very likely true, because D is an instance of the
              general claim B makes.
            </p>

            <p className="leading-relaxed mb-4">
              There is no oracle that declares &ldquo;reality.&rdquo; What
              happens is that new conjectures and evidence enter the market,
              and the community updates on them. D drives B&rsquo;s
              credence from 0.55 toward 0.95. The market&rsquo;s credence
              on B has now moved substantially in the direction your
              evidence predicted. We can score both worlds: the world where
              your evidence had moved B to 0.55 before D arrived (so the
              market was already most of the way there), versus the world
              where B was still at 0.30 when D arrived (so D had to do all
              the work). Using log score &mdash;{' '}
              <InlineMath math="S(p, o) = o \ln(p) + (1 - o) \ln(1 - p)" />{' '}
              &mdash; the market that had your evidence was in a better
              position to absorb D:{' '}
              <InlineMath math="S(0.55, 1) \approx -0.60" /> vs.{' '}
              <InlineMath math="S(0.30, 1) \approx -1.20" />. Your marginal
              improvement: 0.60 log-score units.
            </p>

            <p className="leading-relaxed mb-4">
              <strong>Step 4: Entropy weighting.</strong> You contributed
              when B&rsquo;s entropy was 0.88 bits. Your direct impact on
              B: <InlineMath math="0.88 \times 0.60 = 0.53" />.
            </p>

            <p className="leading-relaxed mb-4">
              <strong>Step 5: Upstream impact.</strong> Your evidence for B
              also moved the credence on A, because the market believes B
              depends on A. Before your paper, A was at 0.50. After, A
              moved to 0.60 &mdash; the market infers that if a high-fidelity
              Cas9 variant works this well, the general conjecture that Cas9
              can be engineered for higher fidelity is more likely. The
              impact on A is calculated the same way: what did the market
              forecast for A&rsquo;s implications before your evidence vs.
              after, scored against what eventually happens? You get credit
              for the upstream improvement too, discounted by the graph
              distance between B and A.
            </p>

            <p className="leading-relaxed">
              <strong>Step 6: Downstream impact (over time).</strong> Later,
              someone creates conjecture C: &ldquo;Base-edited T cells show
              durable engraftment in vivo,&rdquo; which depends on B. When
              evidence for C arrives and propagates back to B, improving
              B&rsquo;s prediction track record, you earn additional impact
              &mdash; because your earlier contribution to B made the whole
              downstream branch of the graph possible. This is the
              downstream direction: future work that builds on your
              conjecture generates impact that flows back to you.
            </p>
          </div>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            How the counterfactual is constructed
          </h3>

          <p className="leading-relaxed">
            The counterfactual &mdash; the world without you &mdash; is the
            credence the market would have had if your evidence had never
            been submitted. In the simplest case, this is just the snapshot
            of credences immediately before your contribution. The market was
            at 0.30, you moved it to 0.55, so the counterfactual is 0.30.
          </p>

          <p className="leading-relaxed">
            This gets more complicated when multiple contributions interact.
            If three people submit evidence on the same day, each moving the
            credence, the ordering matters: the first contributor moved it
            from 0.30 to 0.40, the second from 0.40 to 0.50, the third from
            0.50 to 0.55. Each person&rsquo;s counterfactual is the state
            just before their contribution, so the first contributor gets
            credit for the 0.30-to-0.40 move, the second for 0.40-to-0.50,
            and the third for 0.50-to-0.55. The first contributor moved the
            market the most, but from a higher-entropy starting point. The
            third contributed the least marginal movement.
          </p>

          <p className="leading-relaxed">
            For upstream and downstream impact, the key mechanism
            is <strong>bundling</strong>. When you buy a conjecture, you
            don&rsquo;t just buy that one conjecture &mdash; you
            simultaneously buy all the conjectures you implicitly believe
            as a consequence (see{' '}
            <Link
              to="/platform/hello-world"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Hello World: Bundles
            </Link>
            ). If you buy B and your bundle includes A (because you believe
            B depends on A), then when evidence later validates B, the
            counterfactual for A is clear: without your bundle purchase,
            A would not have received that correlated position. The bundle
            is the mechanism by which your beliefs about dependencies become
            visible to the market, and the aggregate pattern of bundles
            across all participants is what allows the market to infer
            the dependency graph (see{' '}
            <Link
              to="/platform/bayesian-networks"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Bayesian Networks
            </Link>
            ).
          </p>

          {/* --- Conjecture Longevity --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Conjecture Longevity
          </h2>

          <p className="leading-relaxed">
            Not all conjectures sustain productive trading activity for the
            same duration. Some generate decades of active trading and
            evidence submission (general relativity&rsquo;s sub-conjectures
            are still being tested a century later). Others resolve after a
            single observation and go silent. The market should reward
            longevity as a signal of the conjecture&rsquo;s value to the
            knowledge graph.
          </p>

          <p className="leading-relaxed">
            Longevity is an artifact of sustained trade volume: a conjecture
            that people keep trading is one that keeps generating new
            questions, new evidence, and new predictions. The system tracks
            several longevity signals:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Active trading duration.</strong> How long the
              conjecture has sustained non-trivial trade volume since its
              creation. Measured as the span between first and most recent
              trade with activity above a minimum threshold.
            </li>
            <li className="leading-relaxed">
              <strong>Entropy decay rate.</strong> How quickly the
              conjecture&rsquo;s entropy has declined. A conjecture whose
              entropy collapses rapidly was resolved by a single decisive
              piece of evidence. One whose entropy declines slowly is
              sustaining genuine uncertainty over time &mdash; a richer
              source of market activity and information.
            </li>
            <li className="leading-relaxed">
              <strong>Downstream activity span.</strong> How long the
              conjecture&rsquo;s sub-conjectures remain active. A parent
              conjecture whose children are still generating trades is alive
              in the knowledge graph even if its own entropy has reached
              near-zero.
            </li>
          </ul>

          <p className="leading-relaxed">
            Longevity weighting in the scoring system means that holding a
            position in a long-lived, actively traded conjecture contributes
            more to your portfolio value than holding a position in a
            flash-in-the-pan conjecture that resolved immediately. Under
            entropy pricing, dead conjectures naturally stop generating
            reward: once entropy approaches zero, there is no uncertainty
            left to resolve, and new positions earn negligible directional
            return. The cost of entry is near-zero but so is the potential
            upside, which should naturally discourage further purchases. See{' '}
            <Link
              to="/platform/open-problems"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Open Problems
            </Link>{' '}
            for more on this tension.
          </p>

          {/* --- Conjecture Processability --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Conjecture Processability
          </h2>

          <p className="leading-relaxed">
            Not all conjectures are equally useful as market instruments. The
            system tracks several quality dimensions that measure how well
            the market can process a given conjecture. These metrics help
            participants identify dead instruments and focus capital on
            claims that generate information.
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Entropy sensitivity to evidence.</strong> How much does
              the entropy change when new evidence is published? A conjecture
              whose entropy never moves is untradeable &mdash; no observation
              changes anyone&rsquo;s beliefs. Tautological and unfalsifiable
              conjectures score zero on this dimension.
            </li>
            <li className="leading-relaxed">
              <strong>Bid-ask spread width.</strong> A tight spread indicates
              participants agree on what the conjecture means, even if they
              disagree on its truth. A wide spread often signals definitional
              ambiguity rather than genuine uncertainty. Conjectures with
              ambiguous operationalization produce persistently wide spreads.
            </li>
            <li className="leading-relaxed">
              <strong>Trade volume and participant diversity.</strong> High
              volume from diverse participants indicates the conjecture has
              downstream relevance. Low volume or volume concentrated among
              a few participants suggests the conjecture is isolated from the
              broader knowledge graph.
            </li>
            <li className="leading-relaxed">
              <strong>Downstream conjecture count.</strong> Conjectures that
              spawn sub-conjectures are more valuable as market instruments
              because they generate additional trading surfaces and allow
              evidence to propagate through the attribution graph. A
              conjecture with zero downstream connections is a dead end.
            </li>
            <li className="leading-relaxed">
              <strong>Resolution criteria clarity.</strong> Can participants
              agree on what evidence would move the credence to 0.95 or
              0.05? If not, the conjecture is likely ambiguous or
              unfalsifiable. This can be assessed by surveying participants
              on their resolution criteria and measuring agreement.
            </li>
            <li className="leading-relaxed">
              <strong>Entropy contribution to portfolio.</strong> How much
              remaining uncertainty does holding this conjecture add to a
              portfolio? A portfolio full of near-zero-entropy conjectures
              may have many positions but generates no directional reward.
              This metric helps distinguish substantive positions from dead
              weight.
            </li>
          </ul>

          {/* --- Summary --- */}
          <div
            className="rounded-lg border p-6 mt-10"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Summary of key metrics
            </h3>
            <ul className="space-y-3">
              <li className="leading-relaxed">
                <strong>Portfolio value</strong> &mdash; the
                entropy-weighted directional return across all your
                positions, reflecting how far the consensus has moved toward
                you since entry.
              </li>
              <li className="leading-relaxed">
                <strong>Attribution graph position</strong> &mdash; the
                upstream royalties you earn as downstream conjectures build
                on your work. See{' '}
                <Link
                  to="/platform/bayesian-networks"
                  className="underline"
                  style={{ color: 'var(--accent)' }}
                >
                  Bayesian Networks
                </Link>{' '}
                for how the dependency graph is learned from trades.
              </li>
              <li className="leading-relaxed">
                <strong>Calibration</strong> &mdash; your long-run
                directional accuracy across all trades, determining your
                trust weight in the system.
              </li>
              <li className="leading-relaxed">
                <strong>Total conjecture impact</strong> &mdash; the
                cumulative entropy-weighted Shapley-style measure of your
                contributions across the entire conjecture graph.
              </li>
              <li className="leading-relaxed">
                <strong>Conjecture longevity</strong> &mdash; how long a
                conjecture sustains productive trading activity and entropy
                above zero, used to weight its contribution to your veracity
                consensus.
              </li>
              <li className="leading-relaxed">
                <strong>Conjecture processability</strong> &mdash; a quality
                profile measuring entropy sensitivity, spread width, volume,
                downstream count, resolution clarity, and entropy
                contribution to portfolio.
              </li>
            </ul>
          </div>

        </div>
      </div>
    </div>
  )
}
