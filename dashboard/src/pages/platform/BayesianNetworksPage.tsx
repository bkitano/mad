import { Link } from 'react-router-dom'
import { BlockMath, InlineMath } from 'react-katex'

export default function BayesianNetworksPage() {
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
            Bayesian Networks
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            How conjecture dependencies emerge from correlated trades, and
            the challenge of extracting causal structure from market behavior.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* --- The idea --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The idea
          </h2>

          <p className="leading-relaxed">
            Science is not a flat list of independent claims. Conjectures
            depend on other conjectures. &ldquo;CRISPR base editing achieves
            &gt;90% efficiency in T cells&rdquo; depends on &ldquo;Cas9
            variants can be engineered for higher fidelity.&rdquo; &ldquo;Long-context
            transformers scale sub-quadratically&rdquo; depends on
            &ldquo;sparse attention mechanisms preserve downstream task
            performance.&rdquo; These dependencies are not decorative
            metadata &mdash; they are the load-bearing structure of the
            knowledge graph.
          </p>

          <p className="leading-relaxed">
            The ideal is that the market learns this dependency structure
            automatically, without anyone having to declare it. Participants
            reveal dependencies through their trading behavior: if you
            believe B depends on A, you buy both A and B, then publish
            evidence for B. The correlated positions across many participants,
            combined with the pattern of how evidence for one conjecture
            moves the credence of another, should in principle encode a
            Bayesian network &mdash; a directed acyclic graph where edges
            represent conditional dependencies between conjectures.
          </p>

          {/* --- Attribution graphs --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Attribution graphs
          </h2>

          <p className="leading-relaxed">
            The market represents conjecture relationships as a directed
            graph where edges carry <strong>diagnostic weights</strong>{' '}
            &mdash; how much evidence for one conjecture tells you about
            another.
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
            through the graph. If <InlineMath math="B" /> would have been
            likely regardless of whether <InlineMath math="A" /> is true,
            then observing <InlineMath math="B" /> tells you nothing about{' '}
            <InlineMath math="A" />. But if <InlineMath math="B" /> heavily
            depends on <InlineMath math="A" />, then confirming{' '}
            <InlineMath math="B" /> provides strong evidence for{' '}
            <InlineMath math="A" />.
          </p>

          <p className="leading-relaxed">
            Under directed entropy pricing, the upstream propagation is also
            entropy-weighted. If A is at high entropy when evidence for B
            arrives, the information gain for A is larger and the reward for
            A-holders is proportionally greater. If A is already near-certain,
            the same evidence for B barely moves A&rsquo;s entropy and
            generates little additional reward.
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
              <InlineMath math="B" />, a portion of the entropy-weighted
              reward flows upstream to the ancestors of{' '}
              <InlineMath math="B" /> with geometric decay:
            </p>
            <div className="overflow-x-auto">
              <BlockMath math="w_u(B) = \frac{\lambda^{d(u,B)}}{\sum_{v \in \mathrm{Anc}(B)} \lambda^{d(v,B)}}, \qquad 0 < \lambda < 1" />
            </div>
            <p className="leading-relaxed">
              where <InlineMath math="d(u,B)" /> is the graph distance.
              Holders of upstream conjectures earn residuals from downstream
              activity &mdash; founders of genuinely useful ideas are
              rewarded as the research tree grows. The residuals are larger
              when the upstream conjecture is still at high entropy, because
              downstream evidence is doing more work to resolve it.
            </p>
          </div>

          {/* --- From trades to structure --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            From correlated trades to causal structure
          </h2>

          <p className="leading-relaxed">
            The attribution graph above assumes we <em>know</em> the
            dependency structure &mdash; that someone has declared B depends
            on A. But the deeper ambition is to <em>learn</em> the structure
            from market behavior. If many participants hold correlated
            positions across conjectures A and B, and evidence for B
            consistently moves A&rsquo;s credence, the market has implicitly
            encoded the conditional relationship{' '}
            <InlineMath math="P(A \mid B)" />. The question is whether we
            can extract a coherent Bayesian network from this signal.
          </p>

          <p className="leading-relaxed">
            Without explicit structure, the observable data is just a matrix
            of positions: each participant holds some quantity of shares in
            some subset of conjectures, and over time evidence events cause
            credence updates. From this alone, inferring a DAG is extremely
            hard. But the market has a mechanism that makes it much
            easier: <strong>bundles</strong>.
          </p>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Bundles as dependency declarations
          </h3>

          <p className="leading-relaxed">
            When a participant buys a conjecture, they don&rsquo;t buy it
            in isolation &mdash; they buy a{' '}
            <Link
              to="/platform/hello-world"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              bundle
            </Link>{' '}
            of all the conjectures they implicitly believe as a consequence.
            If you buy B, you also buy A, C, and whatever else you think
            must be true for B to hold. The bundle is your belief structure
            made explicit.
          </p>

          <p className="leading-relaxed">
            This is what makes Bayesian network inference tractable. Without
            bundles, the market only sees that participant X holds A and B
            &mdash; it cannot tell whether that reflects a believed
            dependency or just a shared research interest. With bundles, the
            market sees that participant X bought A <em>as part of</em>{' '}
            buying B, which is a direct statement that X believes B depends
            on A.
          </p>

          <p className="leading-relaxed">
            Across many participants, the bundle data produces a rich
            co-occurrence matrix. If 80% of bundles that include B also
            include A, the market has strong evidence that the community
            believes B depends on A. If only 10% of bundles with B include
            C, the dependency is weak or absent. The aggregate pattern of
            bundles is the primary input to structure learning &mdash; it is
            far more informative than position-level co-occurrence alone,
            because it captures <em>intentional</em> belief structure rather
            than incidental overlap.
          </p>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Evidence propagation as validation
          </h3>

          <p className="leading-relaxed">
            Bundles tell the market what participants <em>believe</em> the
            dependency structure is. Evidence propagation tells the market
            whether those beliefs are <em>correct</em>. When evidence for B
            is published and A&rsquo;s credence moves, we observe a
            conditional update:{' '}
            <InlineMath math="P(A \mid \text{evidence for } B) \neq P(A)" />.
            If this update is consistent with the bundle-inferred dependency
            (most bundles linking A and B, and evidence for B reliably
            moving A), the edge is validated. If bundles link A and B but
            evidence for B never moves A, the declared dependency has no
            empirical support.
          </p>

          <p className="leading-relaxed">
            The two signals reinforce each other: bundles provide the
            hypothesized graph structure, and evidence propagation confirms
            or disconfirms each edge. Over time, edges with both strong
            bundle support and consistent propagation become high-confidence
            dependencies. Edges with bundle support but no propagation fade.
            And edges with propagation but no bundle support &mdash;
            conjectures that move together but nobody explicitly linked
            &mdash; are candidates for the market to surface as
            &ldquo;have you considered that these might be related?&rdquo;
          </p>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The structure learning problem
          </h3>

          <p className="leading-relaxed">
            Extracting a Bayesian network from observational data is a
            well-studied problem in machine learning, and it is hard. The
            core challenges apply directly to the conjecture market:
          </p>

          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              1. Correlation is not causation
            </h3>
            <p className="leading-relaxed mb-3">
              Two conjectures may be correlated in holdings and evidence
              propagation without either depending on the other. They might
              share a common ancestor, or they might be correlated because
              the same research group works on both. The market sees
              co-occurrence and co-movement, but the DAG requires directed
              edges. From pure observational data, the best we can recover
              is a Markov equivalence class &mdash; a set of DAGs that are
              indistinguishable given the observed conditional independencies.
            </p>
            <p className="leading-relaxed">
              Interventional data helps: when a participant publishes
              evidence specifically targeting B and we observe A move, that
              is closer to an intervention than a passive observation. The
              market&rsquo;s evidence-submission mechanism is a natural
              source of quasi-interventions that can help break symmetries
              in the equivalence class.
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
              2. The graph is large and sparse
            </h3>
            <p className="leading-relaxed mb-3">
              The number of possible DAGs over{' '}
              <InlineMath math="n" /> conjectures grows
              super-exponentially. With thousands of conjectures, exhaustive
              search is impossible. In practice, the graph must be learned
              incrementally and locally &mdash; when new evidence propagates
              from B to A, the system updates its estimate of the edge
              between them without re-evaluating the entire graph.
            </p>
            <p className="leading-relaxed">
              Sparsity helps: most conjectures are conditionally independent
              of most others. The structure learning algorithm only needs to
              find the relatively small set of edges where genuine
              conditional dependencies exist. Constraint-based methods (like
              the PC algorithm) that prune edges via conditional independence
              tests are a natural fit, but they require enough data per edge
              to distinguish real dependencies from noise.
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
              3. Participants are strategic
            </h3>
            <p className="leading-relaxed mb-3">
              In a standard structure learning setting, the data is generated
              by nature. In the conjecture market, the data is generated by
              participants who have incentives. A participant who believes B
              depends on A may buy both &mdash; but a participant who wants
              the market to <em>think</em> B depends on A (perhaps to
              inflate upstream royalties) may also buy both. The structure
              learning algorithm must be robust to strategic behavior.
            </p>
            <p className="leading-relaxed">
              The entropy-weighted cost of positions helps: creating fake
              correlations across many conjectures requires buying into real
              uncertainty, which is expensive. But the incentive to
              manipulate the perceived graph structure exists whenever
              upstream royalties are at stake, and the system must account
              for it.
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
              4. The graph evolves
            </h3>
            <p className="leading-relaxed">
              Scientific dependencies change as knowledge accumulates. A
              conjecture that once depended on another may become independent
              as new theory provides an alternative foundation. The inferred
              Bayesian network must be a living structure, not a static
              snapshot. Edges should strengthen as evidence propagation
              consistently confirms a dependency, and weaken or disappear
              when the propagation pattern changes. The decay rate of edges
              is itself a design parameter.
            </p>
          </div>

          {/* --- What we can extract --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            What we can realistically extract
          </h2>

          <p className="leading-relaxed">
            Given these challenges, the honest answer is that fully automatic
            Bayesian network recovery from trade data is an aspiration, not
            a near-term deliverable. What we <em>can</em> extract in
            practice, in increasing order of difficulty:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Pairwise correlations.</strong> Which conjectures are
              held together more often than chance predicts? This is a
              co-occurrence matrix, computable directly from position data.
              It gives undirected edges with no causal interpretation, but it
              is a useful starting point for surfacing related conjectures.
            </li>
            <li className="leading-relaxed">
              <strong>Evidence propagation patterns.</strong> When evidence
              for B is published, which other conjectures&rsquo; credences
              move? The direction and magnitude of propagation, aggregated
              over many evidence events, gives empirical estimates of
              conditional dependencies. This is directional (evidence for B
              moves A, but evidence for A may not move B) and is the closest
              the market comes to revealing causal structure without explicit
              declaration.
            </li>
            <li className="leading-relaxed">
              <strong>Participant-declared dependencies.</strong> The simplest
              and most reliable signal: participants explicitly state that B
              depends on A when they create conjectures or submit evidence.
              This is not inferred &mdash; it is declared. The market can
              then validate declared dependencies against observed evidence
              propagation: if someone declares B depends on A but evidence
              for B never moves A, the declared edge has no empirical
              support.
            </li>
            <li className="leading-relaxed">
              <strong>Hybrid structure.</strong> The practical approach is
              likely a hybrid: participants declare the obvious dependencies,
              the market infers additional edges from evidence propagation,
              and the system maintains a graph where declared edges are
              strong priors and inferred edges carry confidence intervals
              that tighten with more data.
            </li>
          </ul>

          {/* --- Implications for the market --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Implications for the market
          </h2>

          <p className="leading-relaxed">
            If the market can learn even approximate dependency structure
            from trades, several things follow:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Automatic evidence propagation.</strong> When evidence
              for B is published, the market can automatically update
              A&rsquo;s credence in proportion to the inferred dependency
              strength, without requiring anyone to manually submit evidence
              for A. The Bayesian network becomes a propagation engine.
            </li>
            <li className="leading-relaxed">
              <strong>Coherence checking.</strong> A Bayesian network implies
              constraints on the joint distribution. If the market&rsquo;s
              credences are inconsistent with the inferred graph (e.g.,{' '}
              <InlineMath math="P(A) = 0.90" /> but{' '}
              <InlineMath math="P(B) = 0.10" /> and the graph says B depends
              heavily on A), the inconsistency is visible and creates an
              arbitrage opportunity for participants who can resolve it.
            </li>
            <li className="leading-relaxed">
              <strong>Research direction signaling.</strong> The graph reveals
              which conjectures are upstream bottlenecks &mdash; high-entropy
              nodes with many downstream dependents. Resolving these would
              have the highest information cascade through the graph. The
              market can surface these as high-value research targets.
            </li>
            <li className="leading-relaxed">
              <strong>Portfolio interpretation.</strong> A participant&rsquo;s
              portfolio is not just a list of positions &mdash; it is a
              subgraph of the Bayesian network. The structure of that
              subgraph reveals whether the participant is making
              coherent, connected bets (buying a conjecture and its
              dependencies) or scattered, independent ones.
            </li>
          </ul>

          <p className="leading-relaxed mt-6">
            The gap between the aspiration (fully automatic DAG recovery) and
            the practical reality (hybrid declared-plus-inferred graphs with
            confidence intervals) is itself an open problem. See{' '}
            <Link
              to="/platform/open-problems"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Open Problems
            </Link>{' '}
            for more on the design challenges involved.
          </p>

        </div>
      </div>
    </div>
  )
}
