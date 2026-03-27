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
            How the conjecture market measures performance: portfolio value, attribution graphs, trade history, and conjecture impact.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* --- Portfolio Value --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Portfolio Value
          </h2>

          <p className="leading-relaxed">
            Your portfolio value is the primary measure of how well your scientific
            judgment tracks with the evolving evidence. It is computed as the sum of
            your positions, each weighted by the current market credence of the
            conjecture:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="V_t = \sum_{A \in \text{Portfolio}} n_i(A) \cdot p_t(A)" />
          </div>

          <p className="leading-relaxed">
            where <InlineMath math="n_i(A)" /> is your share count in conjecture{' '}
            <InlineMath math="A" /> and <InlineMath math="p_t(A)" /> is the
            community&rsquo;s current credence&mdash;a number between 0 and 1.
            As new evidence is published and the community updates its beliefs,
            prices move and your portfolio value moves with them.
          </p>

          <p className="leading-relaxed">
            Unlike traditional prediction markets that settle on binary outcomes,
            portfolio value in the conjecture market is <strong>continuously
            evaluated</strong>. There is no final resolution date. Instead, your
            portfolio appreciates or depreciates as the community&rsquo;s beliefs
            shift over time.
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
              You hold 50 shares of &ldquo;CRISPR base editing achieves &gt;90%
              efficiency in primary human T cells&rdquo; at a current credence of
              0.40, and 30 shares of &ldquo;Long-context transformers scale
              sub-quadratically&rdquo; at 0.65.
            </p>
            <p className="leading-relaxed">
              Your portfolio value is{' '}
              <InlineMath math="50 \times 0.40 + 30 \times 0.65 = 39.50" />. If new
              experimental results push the CRISPR conjecture to 0.55, your portfolio
              rises to{' '}
              <InlineMath math="50 \times 0.55 + 30 \times 0.65 = 47.00" />&mdash;a
              gain driven entirely by evidence, not by trading.
            </p>
          </div>

          {/* --- Portfolio Veracity Consensus --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Portfolio Veracity Consensus
          </h2>

          <p className="leading-relaxed">
            Your <strong>portfolio veracity consensus</strong> is the time series of
            your portfolio value. It captures, in aggregate, how well your bets on
            the truth have held up over time. A consistently rising portfolio
            veracity consensus signals that you have been positioning ahead of the
            evidence&mdash;identifying which conjectures the community would come to
            believe before the evidence arrived.
          </p>

          <p className="leading-relaxed">
            This single curve replaces blunt proxies like h-index or citation count.
            It is visible, verifiable, and independent of institutional affiliation
            or publication venue.
          </p>

          {/* --- Rolling Scores --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Rolling Scores
          </h2>

          <p className="leading-relaxed">
            Because conjectures rarely admit final resolution, the market does not
            pay out on binary settlement. Instead, contributors are scored on how
            much their updates improve future predictive performance. When you move
            the market from <InlineMath math="p_t(A)" /> to{' '}
            <InlineMath math="p_t'(A)" />, your reward depends on whether that move
            is later validated by evidence.
          </p>

          <p className="leading-relaxed">
            Each conjecture <InlineMath math="A" /> carries a set of observable
            implications <InlineMath math="Y_A = \{y_1, \dots, y_n\}" />, each with
            a forecast horizon. Your update is scored against later observations
            using a proper scoring rule:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="\text{Reward}_i(A) = \sum_{y \in Y_A} w_y \left[ S\!\left(p^{(i)}_t(y),\, o_y\right) - S\!\left(p^{(-i)}_t(y),\, o_y\right) \right]" />
          </div>

          <p className="leading-relaxed">
            where <InlineMath math="S" /> is a proper scoring rule such as log
            score, <InlineMath math="p^{(i)}_t(y)" /> is the forecast path with
            your update, and <InlineMath math="p^{(-i)}_t(y)" /> is the
            counterfactual without it. This means you are rewarded for{' '}
            <strong>marginal improvement in calibrated future prediction</strong>,
            not for claiming certainty about <InlineMath math="A" /> itself.
          </p>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Delayed vesting
            </h3>
            <p className="leading-relaxed mb-4">
              To prevent noise from being permanently capitalized, rewards vest over
              multiple horizons:
            </p>
            <div className="overflow-x-auto">
              <BlockMath math="\text{Payout}_i(t) = \sum_{\Delta \in H} \alpha_\Delta \, \text{Reward}_i^{(t+\Delta)}" />
            </div>
            <p className="leading-relaxed">
              with horizons <InlineMath math="H" /> like 3 months, 1 year, and 3
              years. This makes rewards provisional and continuously revised. If
              later evidence reverses your contribution, your payout decreases.
            </p>
          </div>

          {/* --- Attribution Graphs --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Attribution Graphs
          </h2>

          <p className="leading-relaxed">
            Science is not a flat list of independent claims. Conjectures depend on
            other conjectures. The market represents these relationships as a
            directed graph where edges carry <strong>diagnostic weights</strong>&mdash;how
            much evidence for one conjecture tells you about another.
          </p>

          <p className="leading-relaxed">
            If conjecture <InlineMath math="B" /> depends on{' '}
            <InlineMath math="A" /> and other conjectures{' '}
            <InlineMath math="X" />, the system tracks:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="\text{Information contributed to } A \text{ by update on } B \;\propto\; \log \frac{P(B \mid A, X)}{P(B \mid \neg A, X)}" />
          </div>

          <p className="leading-relaxed">
            This likelihood ratio determines how much evidence flows upstream
            through the graph. If <InlineMath math="B" /> would have been likely
            regardless of whether <InlineMath math="A" /> is true, then observing{' '}
            <InlineMath math="B" /> tells you nothing about{' '}
            <InlineMath math="A" />. But if <InlineMath math="B" /> heavily depends
            on <InlineMath math="A" />, then confirming <InlineMath math="B" />{' '}
            provides strong evidence for <InlineMath math="A" />.
          </p>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Upstream royalties
            </h3>
            <p className="leading-relaxed mb-4">
              When someone submits evidence on a downstream conjecture{' '}
              <InlineMath math="B" />, a portion of their submission fee flows
              upstream to the ancestors of <InlineMath math="B" /> with geometric
              decay:
            </p>
            <div className="overflow-x-auto">
              <BlockMath math="w_u(B) = \frac{\lambda^{d(u,B)}}{\sum_{v \in \mathrm{Anc}(B)} \lambda^{d(v,B)}}, \qquad 0 < \lambda < 1" />
            </div>
            <p className="leading-relaxed">
              where <InlineMath math="d(u,B)" /> is the graph distance. Holders of
              upstream conjectures earn residuals from downstream activity&mdash;founders
              of genuinely useful ideas are rewarded as the research tree grows.
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
            Every position change is recorded as a trade. Your trade history is the
            complete log of when you bought or sold positions, at what price, and
            what evidence you attached. This record serves several purposes:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Calibration tracking.</strong> Over many trades, the market can
              assess whether you are well-calibrated&mdash;when you buy at 0.70, are
              the conjectures later confirmed roughly 70% of the time? Long-run
              calibration determines your trust weight in the system.
            </li>
            <li className="leading-relaxed">
              <strong>Timing signal.</strong> Buying early&mdash;before the evidence
              is widely available&mdash;is worth more than buying late. Your trade
              timestamps establish priority.
            </li>
            <li className="leading-relaxed">
              <strong>Staking and evidence.</strong> Each trade that proposes a price
              update requires a stake and attached evidence. The stake limits the
              magnitude of the price move you can propose:
            </li>
          </ul>

          <div className="overflow-x-auto ml-6">
            <BlockMath math="\left|\operatorname{logit}\, p_t'(A) - \operatorname{logit}\, p_t(A)\right| \le c \log(1 + s_j)" />
          </div>

          <p className="leading-relaxed ml-6">
            Large claims require large stakes, making manipulation expensive.
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
              In 1982, Barry Marshall and Robin Warren proposed that stomach ulcers
              were caused by the bacterium <em>Helicobacter pylori</em>, not by
              stress or diet. This was dismissed by nearly all gastroenterologists.
              A conjecture like &ldquo;most gastric ulcers are caused by bacterial
              infection&rdquo; would have been priced near 0.05.
            </p>
            <p className="leading-relaxed">
              Marshall famously drank a petri dish of the bacteria to prove his
              point. In the conjecture market, his early trade&mdash;buying at
              0.05&mdash;and his subsequent evidence submission would be recorded
              with exact timestamps. As antibiotic treatment protocols confirmed the
              link through the late 1980s and 1990s, the price would have risen
              steadily. His trade history would show conviction at the right time,
              backed by evidence that moved the market in a direction later validated
              by the entire field.
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
            Total conjecture impact measures the cumulative effect of your
            contributions across the entire conjecture graph. It combines two
            components: the direct score improvement from your evidence submissions,
            and the indirect value that flows through the attribution graph as
            downstream conjectures build on your work.
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="\text{Impact}_i = \sum_{t,\, y} \alpha_t \, w_y \left[ S\!\left(p^{(i)}_t(y),\, o_y\right) - S\!\left(p^{(-i)}_t(y),\, o_y\right) \right]" />
          </div>

          <p className="leading-relaxed">
            This is a Shapley-style attribution: your impact is the difference
            between the world with your contributions and the counterfactual world
            without them, measured across all affected predictions and time
            horizons. In practice, the system approximates this via sequential
            log-score decomposition.
          </p>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              What high impact looks like
            </h3>
            <ul className="space-y-3">
              <li className="leading-relaxed">
                <strong>Direct evidence.</strong> You submit a replication or
                counter-evidence that significantly moves a conjecture&rsquo;s price,
                and the move is later validated.
              </li>
              <li className="leading-relaxed">
                <strong>Foundational conjectures.</strong> You create a conjecture
                that spawns many downstream conjectures, each generating trade volume
                and evidence. Upstream royalties accrue to you as the tree grows.
              </li>
              <li className="leading-relaxed">
                <strong>Cross-domain connections.</strong> You link conjectures from
                different fields, creating new diagnostic edges in the attribution
                graph that improve predictive performance for both communities.
              </li>
            </ul>
          </div>

          {/* --- Conjecture Equity --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Conjecture Equity
          </h2>

          <p className="leading-relaxed">
            Each conjecture maintains a cap table separate from its credence.
            Credence <InlineMath math="p_t(A)" /> reflects the community&rsquo;s
            belief; equity <InlineMath math="\theta_i^t(A)" /> reflects who has
            contributed value. This separation is important&mdash;if the same token
            both sets belief and captures royalties, the system is too easy to game.
          </p>

          <p className="leading-relaxed">
            When you submit evidence that improves a conjecture&rsquo;s rolling
            score, you are minted new shares proportional to the validated
            information gain:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="m_j(A) = \kappa \max(\Delta S_j(A),\, 0)" />
          </div>

          <p className="leading-relaxed">
            Existing holders are diluted, just as early investors in a company are
            diluted when new investors add value:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="\theta_i^{t+1}(A) = \frac{n_i^t(A) + m_i^{\text{vest}}(A)}{N_t(A) + \sum_k m_k^{\text{vest}}(A)}" />
          </div>

          <p className="leading-relaxed">
            If your evidence later degrades predictive performance, shares do not
            vest and some of your stake can be slashed:
          </p>

          <div className="overflow-x-auto">
            <BlockMath math="\text{slash}_j = \mu \max(-\Delta S_j(A),\, 0)" />
          </div>

          <p className="leading-relaxed">
            This creates a system where knowledge works like equity plus
            royalties&mdash;founders of useful conjectures earn ongoing residuals,
            evidence contributors buy in through demonstrated value, and downstream
            work pays upstream dependencies.
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
            evidence submission (general relativity&rsquo;s sub-conjectures are
            still being tested a century later). Others resolve after a single
            observation and go silent. The market should reward longevity as a
            signal of the conjecture&rsquo;s value to the knowledge graph.
          </p>

          <p className="leading-relaxed">
            Longevity is an artifact of sustained trade volume: a conjecture
            that people keep trading is one that keeps generating new questions,
            new evidence, and new predictions. The system tracks several
            longevity signals:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Active trading duration.</strong> How long the conjecture
              has sustained non-trivial trade volume since its creation.
              Measured as the span between first and most recent trade with
              activity above a minimum threshold.
            </li>
            <li className="leading-relaxed">
              <strong>Price sensitivity window.</strong> How long the
              conjecture&rsquo;s price has remained responsive to new evidence.
              A conjecture whose price stopped moving years ago has ended its
              productive life, even if occasional trades still occur.
            </li>
            <li className="leading-relaxed">
              <strong>Downstream activity span.</strong> How long the
              conjecture&rsquo;s sub-conjectures remain active. A parent
              conjecture whose children are still generating trades is alive
              in the knowledge graph even if its own price has stabilized.
            </li>
          </ul>

          <p className="leading-relaxed">
            Longevity weighting in the scoring system means that holding a
            position in a long-lived, actively traded conjecture contributes
            more to your veracity consensus than holding a position in a
            flash-in-the-pan conjecture that resolved immediately. The open
            question is how to handle conjectures that are resolved after a
            single observation (e.g., &ldquo;the paper had 12 authors&rdquo;)
            &mdash; what disincentivizes participants from continuing to buy
            shares of a dead conjecture? The price itself may be the answer:
            once a conjecture converges to ~1.0 or ~0.0, the potential return
            from additional shares is near zero, which should naturally
            discourage further purchases. But this only works if shares have
            a meaningful cost. See{' '}
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
            system tracks several quality dimensions that measure how well the
            market can process a given conjecture. These metrics help participants
            identify dead instruments and focus capital on claims that generate
            information.
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Price sensitivity to evidence.</strong> How much does the
              price move when new evidence is published? A conjecture with zero
              price sensitivity is untradeable&mdash;no observation changes
              anyone&rsquo;s beliefs. Tautological and unfalsifiable conjectures
              score zero on this dimension.
            </li>
            <li className="leading-relaxed">
              <strong>Bid-ask spread width.</strong> A tight spread indicates
              participants agree on what the conjecture means, even if they
              disagree on its truth. A wide spread often signals definitional
              ambiguity rather than genuine uncertainty. Conjectures with
              ambiguous operationalization produce persistently wide spreads.
            </li>
            <li className="leading-relaxed">
              <strong>Trade volume and participant diversity.</strong> High volume
              from diverse participants indicates the conjecture has downstream
              relevance. Low volume or volume concentrated among a few
              participants suggests the conjecture is isolated from the broader
              knowledge graph.
            </li>
            <li className="leading-relaxed">
              <strong>Downstream conjecture count.</strong> Conjectures that spawn
              sub-conjectures are more valuable as market instruments because they
              generate additional trading surfaces and allow evidence to propagate
              through the attribution graph. A conjecture with zero downstream
              connections is a dead end.
            </li>
            <li className="leading-relaxed">
              <strong>Resolution criteria clarity.</strong> Can participants agree
              on what evidence would move the price to 0.95 or 0.05? If not, the
              conjecture is likely ambiguous or unfalsifiable. This can be
              assessed by surveying participants on their resolution criteria and
              measuring agreement.
            </li>
            <li className="leading-relaxed">
              <strong>Portfolio delta contribution.</strong> How much sensitivity
              to new evidence does holding this conjecture add to a portfolio? A
              portfolio full of trivially true conjectures (price ~1.0, zero
              delta) may look large but generates no information. This metric
              helps distinguish substantive positions from dead weight.
            </li>
          </ul>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Conjecture quality scorecard
            </h3>
            <p className="leading-relaxed mb-4">
              Each conjecture can be scored across these dimensions to produce a
              processability profile. For example, a tautological conjecture like
              &ldquo;In efficient markets, all available information is reflected
              in asset prices&rdquo; would score: price sensitivity 0/10, spread
              width 9/10 (wide), volume 1/10, downstream count 0, resolution
              clarity 0/10, portfolio delta 0. A well-formed conjecture like
              &ldquo;Transformer models with fewer than 1B parameters can achieve
              &gt;90% accuracy on MATH&rdquo; would score: price sensitivity
              8/10, spread width 2/10 (tight), volume 7/10, downstream count 4+,
              resolution clarity 9/10, portfolio delta 7/10.
            </p>
            <p className="leading-relaxed">
              See{' '}
              <Link
                to="/platform/example-conjectures"
                className="underline"
                style={{ color: 'var(--accent)' }}
              >
                Example Conjectures
              </Link>{' '}
              for worked examples showing how different conjecture types score on
              these dimensions.
            </p>
          </div>

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
                <strong>Portfolio value</strong> &mdash; the mark-to-market worth of
                all your positions, updated continuously as credences shift.
              </li>
              <li className="leading-relaxed">
                <strong>Portfolio veracity consensus</strong> &mdash; the time series
                of your portfolio value, revealing your track record of scientific
                judgment.
              </li>
              <li className="leading-relaxed">
                <strong>Rolling score</strong> &mdash; your marginal contribution to
                predictive accuracy, measured against proper scoring rules with
                delayed vesting.
              </li>
              <li className="leading-relaxed">
                <strong>Attribution graph position</strong> &mdash; the upstream
                royalties you earn as downstream conjectures build on your work.
              </li>
              <li className="leading-relaxed">
                <strong>Calibration</strong> &mdash; your long-run forecast accuracy
                across all trades, determining your trust weight in the system.
              </li>
              <li className="leading-relaxed">
                <strong>Total conjecture impact</strong> &mdash; the cumulative
                Shapley-style measure of your contributions across the entire
                conjecture graph.
              </li>
              <li className="leading-relaxed">
                <strong>Conjecture equity</strong> &mdash; your ownership stake in
                individual conjectures, earned through validated evidence and subject
                to dilution and slashing.
              </li>
              <li className="leading-relaxed">
                <strong>Conjecture longevity</strong> &mdash; how long a conjecture
                sustains productive trading activity, used to weight its contribution
                to your veracity consensus.
              </li>
              <li className="leading-relaxed">
                <strong>Conjecture processability</strong> &mdash; a quality profile
                measuring price sensitivity, spread width, volume, downstream count,
                resolution clarity, and portfolio delta contribution.
              </li>
            </ul>
          </div>

        </div>
      </div>
    </div>
  )
}
