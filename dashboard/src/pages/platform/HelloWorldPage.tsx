import { Link } from 'react-router-dom'

export default function HelloWorldPage() {
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
            Hello, World
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Portfolios, positions, and portfolio veracity consensus.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>
          <p className="leading-relaxed">
            Every participant in the conjecture market has a{' '}
            <strong>portfolio</strong>: a collection of positions in
            conjectures they have bought or sold. Each conjecture has a{' '}
            <strong>credence</strong> &mdash; a number between 0 and 1
            reflecting the community&rsquo;s aggregate belief in whether it
            is true.
          </p>

          <p className="leading-relaxed">
            The <strong>price</strong> of taking a position is not the
            credence itself. It is related to the conjecture&rsquo;s
            current <strong>entropy</strong> &mdash; a measure of how
            uncertain the market is. When the credence is near 0.50
            (maximum uncertainty), entropy is high and positions are
            expensive: you are buying into genuine unresolved doubt. When
            the credence is near 0 or 1 (the community has largely made up
            its mind), entropy is low and positions are cheap: there is
            little uncertainty left to trade.
          </p>

          <p className="leading-relaxed">
            Your portfolio value tracks how well your scientific judgment
            aligns with the evidence as it unfolds. It is not simply shares
            times credence &mdash; it depends on <em>when</em> you entered
            (the entropy at the time of your purchase) and <em>whether the
            consensus moved toward your position</em>. See{' '}
            <Link
              to="/platform/market-incentives"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Market Incentives
            </Link>{' '}
            for the full framework.
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
              Suppose you are a researcher studying protein folding. The
              conjecture &ldquo;AlphaFold-class models generalize to
              disordered proteins&rdquo; has a credence of 0.35. Entropy is
              fairly high &mdash; the community is uncertain. You buy a YES
              position, paying an entropy-derived cost.
            </p>
            <p className="leading-relaxed mb-4">
              Over the next few months, two independent groups publish
              results showing strong performance on disordered protein
              benchmarks. The credence rises to 0.62. Your portfolio value
              increases &mdash; you entered at high entropy and the
              consensus moved toward your position.
            </p>
            <p className="leading-relaxed">
              Conversely, if a replication attempt fails and the credence
              drops to 0.20, your position loses value: the consensus moved
              away from you. The market doesn&rsquo;t care about your
              credentials or intent &mdash; only whether you bore
              uncertainty in the direction the evidence eventually supported.
            </p>
          </div>

          <p className="leading-relaxed">
            This is the fundamental unit of participation: you hold
            conjectures, and your portfolio value tracks how well your
            scientific judgment aligns with the evidence as it unfolds.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Making a trade
          </h2>

          <p className="leading-relaxed">
            When you buy or sell a conjecture, you are making a <strong>
            trade</strong>. A trade has two optional attachments:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Evidence.</strong> You can attach the reasoning or
              data that motivated your trade &mdash; a paper, a dataset,
              an argument, a link to an experimental result. This is not
              required. You can trade on pure intuition. But attaching
              evidence is how the market learns <em>why</em> credences
              should move, not just that someone thinks they should.
            </li>
            <li className="leading-relaxed">
              <strong>Visibility.</strong> Both your trade and your evidence
              can independently be set to <strong>public</strong> or{' '}
              <strong>private</strong>. This creates four combinations:
            </li>
          </ul>

          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <ul className="space-y-3">
              <li className="leading-relaxed">
                <strong>Public trade, public evidence.</strong> Everyone can
                see that you bought the conjecture and why. This is the most
                informative signal to the market &mdash; it moves credence
                and gives the community a reason to update. This is what you
                do when you want to convince people.
              </li>
              <li className="leading-relaxed">
                <strong>Public trade, private evidence.</strong> Everyone can
                see you took a position, but not why. The market knows
                someone with skin in the game believes something, but
                can&rsquo;t evaluate the reasoning. This is useful when your
                evidence is proprietary or pre-publication and you want to
                establish priority without revealing the details.
              </li>
              <li className="leading-relaxed">
                <strong>Private trade, public evidence.</strong> You publish
                evidence that should move the credence, but no one knows
                you&rsquo;re positioned to benefit. This separates the
                intellectual contribution from the financial position. The
                evidence stands on its own merits.
              </li>
              <li className="leading-relaxed">
                <strong>Private trade, private evidence.</strong> A silent
                position. The market registers the trade&rsquo;s effect on
                credence (if it uses an AMM, the credence still moves), but
                no one knows who traded or why. This is the default for
                participants who want to accumulate positions without
                signaling.
              </li>
            </ul>
          </div>

          <p className="leading-relaxed mt-4">
            The visibility choices create different incentive dynamics.
            Public evidence with a public trade is the strongest signal and
            typically moves the credence the most, because the community can
            evaluate both the claim and the reasoning. A private trade with
            no evidence still affects the credence through the pricing
            mechanism, but gives the community nothing to update on beyond
            the price movement itself.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Bundles
          </h2>

          <p className="leading-relaxed">
            When you buy a conjecture, you rarely believe it in isolation.
            If you believe &ldquo;AlphaFold-class models generalize to
            disordered proteins,&rdquo; you probably also believe a set of
            upstream conjectures that make it plausible: &ldquo;attention
            mechanisms can capture long-range residue interactions,&rdquo;
            &ldquo;disordered proteins have learnable structural
            regularities,&rdquo; and so on. These beliefs are implicit in
            your purchase of the original conjecture.
          </p>

          <p className="leading-relaxed">
            The market makes this explicit through <strong>bundles</strong>.
            When you buy a conjecture, you also buy all the conjectures you
            believe as a consequence &mdash; the upstream dependencies that
            make your target conjecture coherent. This is not optional
            metadata; it is the core mechanism by which the market learns
            what depends on what.
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
              You believe conjecture B: &ldquo;CRISPR base editing achieves
              &gt;90% efficiency in primary human T cells.&rdquo; When you
              buy B, you also buy conjecture A: &ldquo;Cas9 variants can
              be engineered for higher fidelity&rdquo; &mdash; because
              if A is false, B is much less likely. Your bundle is {'{'}A,
              B{'}'}.
            </p>
            <p className="leading-relaxed mb-4">
              Another participant buys B but bundles it with C: &ldquo;Lipid
              nanoparticle delivery achieves sufficient nuclear
              uptake.&rdquo; Their bundle is {'{'}B, C{'}'}. A third
              participant buys A, B, and C together.
            </p>
            <p className="leading-relaxed">
              The market now has three bundles. The overlap tells it
              something: A and B are correlated (two out of three bundles
              contain both), B and C are correlated (two out of three), and
              A and C are weakly correlated (one out of three &mdash; only
              through the third participant). As more participants trade and
              more bundles accumulate, the co-occurrence matrix becomes a
              richer and richer signal of which conjectures the community
              believes are logically connected.
            </p>
          </div>

          <p className="leading-relaxed">
            Bundles serve two purposes simultaneously. First, they ensure
            that when you take a position, you are expressing your full
            belief structure &mdash; not just an isolated opinion on one
            claim, but the web of claims you think must be true for that
            one claim to hold. Second, the aggregate pattern of bundles
            across all participants is the raw data from which the market
            infers its dependency graph (see{' '}
            <Link
              to="/platform/bayesian-networks"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Bayesian Networks
            </Link>
            ). No one declares the graph. It emerges from the overlapping
            structure of what people buy together.
          </p>

          <p className="leading-relaxed">
            This also means your reward is not just about one conjecture.
            When evidence validates B and drives its credence up, your
            position in A also benefits &mdash; because you bundled them
            together, expressing a belief that they are linked. If the
            market agrees (because many other bundles also link A and B),
            the credence update propagates. You are rewarded not just for
            being right about B, but for correctly identifying that A and B
            are connected.
          </p>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Historical example: Continental drift
            </h3>
            <p className="leading-relaxed mb-4">
              In 1912, Alfred Wegener proposed that the continents were once joined and had
              drifted apart. The idea was dismissed by most geologists for decades. If a
              conjecture market had existed, &ldquo;the continents move relative to one
              another&rdquo; might have been priced at 0.05&mdash;a fringe claim.
            </p>
            <p className="leading-relaxed mb-4">
              Wegener would have bought a position. Over the following decades, evidence
              accumulated: matching fossil records across oceans, mid-ocean ridge
              discoveries in the 1950s, and magnetic striping on the seafloor in the 1960s.
              Each piece of evidence would have moved the price upward. By the time plate
              tectonics was widely accepted in the late 1960s, the price would have been
              near 0.95.
            </p>
            <p className="leading-relaxed">
              Wegener&rsquo;s portfolio would have appreciated enormously&mdash;not because
              anyone declared him right, but because the community&rsquo;s credence moved in
              the direction he positioned for, over fifty years of accumulating evidence.
              Meanwhile, a geologist who held positions in &ldquo;continents are
              fixed&rdquo; would have watched their portfolio decline as the evidence mounted.
            </p>
          </div>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              What your portfolio tells you
            </h3>
            <ul className="space-y-3">
              <li className="leading-relaxed">
                <strong>Rising portfolio value</strong> means your positions are being
                validated by new evidence. The conjectures you believed in are gaining
                credence.
              </li>
              <li className="leading-relaxed">
                <strong>Falling portfolio value</strong> means the evidence is moving against
                your positions. Time to re-evaluate your beliefs, or double down if you think
                the market is wrong.
              </li>
              <li className="leading-relaxed">
                <strong>Flat portfolio value</strong> means the conjectures you hold are
                stable&mdash;no significant new evidence has shifted the community&rsquo;s
                beliefs in either direction.
              </li>
            </ul>
          </div>

          <div
            className="rounded-lg border p-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-4"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Historical example: Barbara McClintock and transposable elements
            </h3>
            <p className="leading-relaxed mb-4">
              In the 1940s and 50s, Barbara McClintock discovered &ldquo;jumping
              genes&rdquo;&mdash;segments of DNA that move within a genome. Her work was
              largely ignored or dismissed by the genetics community for decades. A
              conjecture like &ldquo;genes can change position within chromosomes&rdquo;
              would have been priced very low.
            </p>
            <p className="leading-relaxed">
              McClintock&rsquo;s portfolio would have been flat for years&mdash;no new
              evidence moved the price, because few people were looking. But when molecular
              biology tools matured in the 1970s and transposons were independently confirmed
              in bacteria, the price would have surged. She won the Nobel Prize in 1983. A
              flat portfolio is not necessarily a wrong portfolio&mdash;it may simply be
              waiting for the right evidence to arrive.
            </p>
          </div>

          <p className="leading-relaxed">
            Over time, a consistently appreciating portfolio is the market&rsquo;s way of
            saying you have good scientific intuition. You identified which conjectures
            would be borne out before the evidence arrived. That track record is visible,
            verifiable, and independent of institutional affiliation or publication count.
          </p>
        </div>
      </div>
    </div>
  )
}
