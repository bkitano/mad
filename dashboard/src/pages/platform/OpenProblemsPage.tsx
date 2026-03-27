import { Link } from 'react-router-dom'
import { BlockMath, InlineMath } from 'react-katex'

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
              The trivially-true portfolio problem
            </h2>
            <p className="leading-relaxed mb-4">
              A participant can accumulate a large portfolio of technically true but
              uninformative conjectures. Consider someone who holds positions in dozens of
              correlation-based claims (&ldquo;countries with higher chocolate consumption
              produce more Nobel laureates&rdquo;), already-settled facts (&ldquo;water
              boils at 100&deg;C&rdquo;), and narrowly confirmed measurements
              (&ldquo;GPT-4 scores 86.4% on MMLU&rdquo;). Every position sits at high
              credence. The portfolio looks impressive on paper&mdash;many conjectures, all
              priced near 1.0.
            </p>
            <p className="leading-relaxed mb-4">
              But this portfolio generates essentially zero information. It has no delta
              (sensitivity to new evidence), no alpha (returns from insight), and no
              downstream activity (no sub-conjectures are spawned). The participant&rsquo;s
              veracity consensus flatlines despite the large position count. They occupy
              market capacity without contributing to the knowledge graph.
            </p>
            <p className="leading-relaxed mb-4">
              This is a problem because the market&rsquo;s incentive structure should
              reward genuine scientific judgment, not the accumulation of trivially
              confirmable claims. Several approaches are worth exploring:
            </p>
            <ul className="space-y-4">
              <li className="leading-relaxed">
                <strong>Portfolio delta weighting.</strong> Weight a participant&rsquo;s
                overall score not just by portfolio value, but by portfolio delta&mdash;how
                much their portfolio value would change if new evidence arrived. A portfolio
                of settled claims has zero delta and would receive proportionally less
                credit, even if its absolute value is high.
              </li>
              <li className="leading-relaxed">
                <strong>Information contribution scoring.</strong> Only count positions
                where the participant&rsquo;s entry moved the market or where their
                evidence submission improved the rolling score. Holding a conjecture that
                was already at 0.99 when you bought it contributes nothing to the
                market&rsquo;s information content.
              </li>
              <li className="leading-relaxed">
                <strong>Conjecture processability filtering.</strong> Use the processability
                metrics (price sensitivity, spread width, downstream count, etc.) to
                discount or exclude low-quality conjectures from portfolio scoring entirely.
                A position in a conjecture with zero price sensitivity and no downstream
                conjectures simply would not count toward your veracity consensus.
              </li>
              <li className="leading-relaxed">
                <strong>Opportunity cost mechanisms.</strong> If capital deployed in
                trivially true conjectures could otherwise be deployed in informative ones,
                the market could impose a carrying cost on low-activity positions. This
                would make it economically irrational to park capital in dead instruments,
                naturally discouraging the strategy.
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
                The correlation portfolio
              </h3>
              <p className="leading-relaxed mb-4">
                Messerli (2012) published a correlation between national chocolate
                consumption and Nobel laureates per capita. The correlation is real
                (<em>r</em> = 0.791). A participant who buys this conjecture at 0.80 is
                technically correct and will hold a stable position. But the conjecture has
                no causal content&mdash;increasing chocolate distribution would not increase
                Nobel production. The real drivers are wealth, education infrastructure, and
                research funding, none of which appear in the conjecture.
              </p>
              <p className="leading-relaxed">
                Now imagine a participant who builds an entire portfolio of such
                correlations. Every position is technically true. The portfolio is large.
                But it has generated zero predictions, zero sub-conjectures, and zero
                insights about the world. The open problem is how to make this visible in
                the scoring system&mdash;so that the market rewards participants who hold
                conjectures that <em>matter</em>, not just conjectures that happen to be
                true.
              </p>
            </div>

            <p className="leading-relaxed mt-6">
              The risk of not solving this is a market that can be gamed by accumulating
              trivially confirmable claims. The risk of over-solving it is penalizing
              participants for holding settled conjectures that serve as legitimate anchors
              in the knowledge graph. Settled conjectures have real value as
              reference points&mdash;the goal is to distinguish between &ldquo;settled and
              structurally important&rdquo; and &ldquo;settled and informationally dead.&rdquo;
            </p>
          </section>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              What does anything cost?
            </h2>
            <p className="leading-relaxed mb-4">
              The conjecture market as described does not have a clear answer to
              a basic question: what are positions denominated in? In financial
              prediction markets, you buy shares with dollars. In the conjecture
              market, participants are expressing beliefs about scientific claims.
              It&rsquo;s not obvious that dollars are the right unit, or that any
              monetary unit should be involved at all.
            </p>
            <p className="leading-relaxed mb-4">
              Having an opinion is free. Anyone can look at a conjecture and
              think &ldquo;that&rsquo;s probably true&rdquo; or &ldquo;that&rsquo;s
              probably false.&rdquo; The question is what it costs to
              have your opinion <em>affect the market price</em>. If it costs
              nothing, then anyone can take an arbitrarily large position on
              anything, and the price becomes a popularity contest rather than a
              credence signal. If it costs too much, the market excludes
              participants who have genuine knowledge but no capital.
            </p>
            <p className="leading-relaxed mb-4">
              Several possibilities:
            </p>
            <ul className="space-y-4">
              <li className="leading-relaxed">
                <strong>Free first share, paid additional shares.</strong> Every
                participant gets one share of any conjecture for free &mdash; this
                is the &ldquo;having an opinion is free&rdquo; baseline. But if
                you want your opinion to count <em>more</em> than the average
                person&rsquo;s, you must pay for additional shares. This creates a
                natural cost function: expressing a belief costs nothing, but
                expressing conviction that your belief is <em>more informed than
                the average participant&rsquo;s</em> costs something. The payment
                could be in a market currency, reputation points, or staked
                conviction from other positions.
              </li>
              <li className="leading-relaxed">
                <strong>Reputation-weighted influence.</strong> Instead of
                paying per share, each participant&rsquo;s position is weighted
                by their track record. A participant with high calibration and
                veracity consensus has more influence per share than a newcomer.
                This avoids the capital problem (expertise matters, not wealth)
                but creates a bootstrapping problem: how do new participants
                build reputation?
              </li>
              <li className="leading-relaxed">
                <strong>Staked conviction.</strong> Taking a larger position on
                one conjecture requires staking value from your other positions.
                This creates a natural constraint: you can&rsquo;t claim to be
                highly confident in everything simultaneously, because confidence
                in one claim comes at the cost of exposure elsewhere. Your total
                conviction budget is finite even if any individual opinion is
                free.
              </li>
              <li className="leading-relaxed">
                <strong>Dollar-denominated.</strong> Participants buy shares
                with real money, just like a prediction market. This aligns
                incentives sharply (people are careful with real money) but
                excludes participants without capital, which could systematically
                bias the market toward the beliefs of wealthy institutions over
                individual researchers.
              </li>
            </ul>
            <p className="leading-relaxed mt-4">
              The deeper question is whether the market needs a constraint
              mechanism at all, or whether one emerges naturally. If there is no
              cost to taking a position, then the market needs some other way to
              prevent participants from taking unlimited positions to sway the
              price. If there is a cost, the market needs to determine what that
              cost is denominated in, and whether the unit can be converted to
              and from real-world value (dollars, academic credit, etc.).
            </p>
          </section>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Counterparty mechanics
            </h2>
            <p className="leading-relaxed mb-4">
              In a traditional market, every trade requires a counterparty:
              someone who buys when you sell, and vice versa. The price is where
              buyer and seller agree. But the conjecture market does not
              necessarily have a fixed supply of shares. If anyone can create a
              new share of a conjecture by expressing an opinion, then there is
              no scarcity of shares and no need for a counterparty to provide
              them.
            </p>
            <p className="leading-relaxed mb-4">
              This raises several questions:
            </p>
            <ul className="space-y-4">
              <li className="leading-relaxed">
                <strong>What does it mean to short?</strong> In a market with
                unlimited share supply, shorting cannot mean &ldquo;borrow and
                sell shares you don&rsquo;t own.&rdquo; It must mean something
                else &mdash; perhaps registering a negative position that pushes
                the price down (see{' '}
                <Link
                  to="/platform/shorting"
                  className="underline"
                  style={{ color: 'var(--accent)' }}
                >
                  Shorting
                </Link>
                ). But if negative positions are free to create, what prevents
                one participant from creating an unbounded number of them?
              </li>
              <li className="leading-relaxed">
                <strong>Where does the money come from?</strong> If a
                participant takes a position at price 0.40 and the price rises
                to 0.80, they&rsquo;ve &ldquo;made money.&rdquo; But who pays?
                In a counterparty market, the people who sold at 0.40 are on the
                other side. In an AMM-based market with no counterparties, the
                market maker (the algorithm) absorbs the loss. This means the
                system needs a subsidy &mdash; someone funds the market maker&rsquo;s
                maximum loss. Who, and why?
              </li>
              <li className="leading-relaxed">
                <strong>Does the market need counterparties at all?</strong> If
                the goal is to aggregate beliefs rather than to facilitate
                exchange, perhaps the market is better modeled as a weighted
                poll with incentive alignment rather than as a trading venue.
                The{' '}
                <Link
                  to="/platform/price-determination"
                  className="underline"
                  style={{ color: 'var(--accent)' }}
                >
                  pricing mechanism
                </Link>{' '}
                could be a function that simply takes all positions as input and
                outputs a credence, without any notion of buying from or selling
                to another participant. This would simplify the system but might
                lose some of the information-revealing properties that emerge
                from genuine exchange.
              </li>
            </ul>
          </section>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Price, longevity, and the resolved-conjecture problem
            </h2>
            <p className="leading-relaxed mb-4">
              Some conjectures are essentially resolved after a single
              observation. &ldquo;The next AlphaFold paper will have more than
              12 authors&rdquo; either happens or it doesn&rsquo;t. Once the
              paper is published, the price converges to ~1.0 or ~0.0 and
              there is nothing left to learn. Yet participants could, in
              principle, continue purchasing shares of this conjecture forever.
            </p>
            <p className="leading-relaxed mb-4">
              The market should reward the <strong>longevity</strong> of a
              conjecture &mdash; how long it remains a productive trading
              surface with active volume and price sensitivity to new evidence.
              A conjecture that generates decades of trading activity (like
              general relativity) is more valuable to the knowledge graph than
              one that resolves in a single observation. But how does the
              market disincentivize participants from continuing to buy shares
              of a dead conjecture?
            </p>
            <p className="leading-relaxed mb-4">
              The price itself is supposed to serve both functions: it
              communicates consensus <em>and</em> it incentivizes or
              disincentivizes purchases. When a conjecture is at 0.99, the
              potential upside from buying is tiny (at most 0.01 per share),
              so rational participants should stop buying. But this only works
              if there is a meaningful cost to purchasing shares. If shares are
              free, there is no disincentive at all. This is another argument
              for some form of position cost (see &ldquo;What does anything
              cost?&rdquo; above).
            </p>
            <p className="leading-relaxed mb-4">
              A related question: should the market track and display conjecture
              longevity as a metric? If so, what counts? Possible measures:
            </p>
            <ul className="space-y-4">
              <li className="leading-relaxed">
                <strong>Active trading duration.</strong> How long the conjecture
                has sustained non-trivial trade volume. A conjecture that had
                100 trades in one day and then went silent is less &ldquo;long-lived&rdquo;
                than one with steady trading over years.
              </li>
              <li className="leading-relaxed">
                <strong>Price sensitivity window.</strong> How long the price
                remained sensitive to new evidence. Once the price stops
                responding to new results, the conjecture&rsquo;s productive
                life is over even if trades continue.
              </li>
              <li className="leading-relaxed">
                <strong>Downstream activity span.</strong> How long the
                conjecture&rsquo;s sub-conjectures remain active. A parent
                conjecture whose children are still generating trades is still
                alive in the knowledge graph, even if the parent&rsquo;s own
                price has stabilized.
              </li>
            </ul>
          </section>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Free opinions and weighted influence
            </h2>
            <p className="leading-relaxed mb-4">
              Having an opinion is free. Anyone can believe anything. The market
              becomes interesting at the point where a participant wants their
              opinion to <em>affect the price</em> more than the average
              person&rsquo;s opinion. This is a statement about self-assessed
              expertise: &ldquo;I think my belief about this conjecture is more
              informed than the typical participant&rsquo;s.&rdquo;
            </p>
            <p className="leading-relaxed mb-4">
              The market needs to price this self-assessment effectively. If
              influence is free, people will claim more influence than they
              deserve, and the market becomes a shouting match. If influence is
              too expensive, the market silences knowledgeable participants who
              can&rsquo;t afford to buy their way in.
            </p>
            <p className="leading-relaxed mb-4">
              One natural structure: everyone gets one &ldquo;vote&rdquo;
              (one share) for free. Additional influence requires staking
              something of value &mdash; reputation, conviction from other
              positions, or currency. The cost of going from 1 share to 10
              shares should reflect the claim &ldquo;my opinion is 10x more
              informed than average.&rdquo; If you&rsquo;re right, you earn
              returns that justify the stake. If you&rsquo;re wrong, you lose
              the stake, which is the market&rsquo;s way of saying &ldquo;your
              opinion was not, in fact, more informed.&rdquo;
            </p>
            <p className="leading-relaxed">
              This creates an interesting connection to the scoring system: a
              participant&rsquo;s track record (calibration, veracity consensus)
              could serve as implicit collateral. A well-calibrated participant
              has earned the right to cheap influence because the market has
              evidence that their opinions are worth more. A poorly calibrated
              participant faces higher costs because their track record suggests
              their opinions are noise.
            </p>
          </section>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Information content and bit potential
            </h2>
            <p className="leading-relaxed mb-4">
              Not all conjectures encode the same amount of information. A
              conjecture at price 0.50 is maximally uncertain &mdash; the market
              has no idea whether it&rsquo;s true or false. If it moves to 0.95,
              that movement encodes a lot of information: the market went from
              maximum uncertainty to near-certainty. But a conjecture that moves
              from 0.94 to 0.95 has barely changed the market&rsquo;s state of
              knowledge.
            </p>
            <p className="leading-relaxed mb-4">
              The <strong>bit potential</strong> of a conjecture is the maximum
              amount of information (in bits) that the market could learn from
              its resolution. For a binary conjecture at price{' '}
              <InlineMath math="p" />, the remaining uncertainty is the binary
              entropy:
            </p>
            <div className="overflow-x-auto">
              <BlockMath math="H(p) = -p \log_2 p - (1-p) \log_2 (1-p)" />
            </div>
            <p className="leading-relaxed mb-4">
              A conjecture at 0.50 has 1 bit of entropy &mdash; maximum
              information potential. A conjecture at 0.99 has about 0.08 bits
              &mdash; almost everything is already known. A conjecture at 1.0
              has 0 bits &mdash; nothing left to learn.
            </p>
            <p className="leading-relaxed mb-4">
              This connects directly to the predictive capacity rubric: a
              conjecture with low bit potential (price near 0 or 1) cannot
              generate much information regardless of whether it&rsquo;s
              &ldquo;right.&rdquo; It&rsquo;s already priced in. The conjectures
              that matter most for the knowledge graph are those with high bit
              potential &mdash; significant uncertainty that, when resolved,
              would tell the community something it doesn&rsquo;t already know.
            </p>
            <p className="leading-relaxed mb-4">
              Bit potential could serve as an impact measure: a participant
              whose portfolio is concentrated in high-entropy conjectures is
              working on the frontier of what the market doesn&rsquo;t know. A
              participant whose portfolio is concentrated in low-entropy
              conjectures (trivially true or already resolved) is not
              contributing to the market&rsquo;s information growth, regardless
              of their portfolio size.
            </p>
            <p className="leading-relaxed">
              The open question is whether the market should explicitly track
              and display bit potential, and whether it should factor into
              scoring. If a participant&rsquo;s total impact is weighted by the
              bit potential of the conjectures they trade, the market naturally
              rewards work on uncertain, frontier questions over accumulation
              of settled claims. But this also means that once a conjecture is
              &ldquo;mostly resolved&rdquo; (price near 0 or 1), there&rsquo;s
              little reward for holding it &mdash; which is exactly the
              behavior we want.
            </p>
          </section>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Uncertainty-weighted pricing
            </h2>
            <p className="leading-relaxed mb-4">
              The standard LMSR cost function (see{' '}
              <Link
                to="/platform/price-determination"
                className="underline"
                style={{ color: 'var(--accent)' }}
              >
                Price Determination
              </Link>
              ) charges the same amount for a share regardless of where the
              price sits. But there is a case for making the cost of a position
              proportional to the conjecture&rsquo;s uncertainty &mdash;
              specifically, proportional to{' '}
              <InlineMath math="|0.5 - p|" />, where{' '}
              <InlineMath math="p" /> is the current price.
            </p>
            <p className="leading-relaxed mb-4">
              Under this scheme, it would be <strong>cheap to buy high-consensus
              conjectures</strong> (price near 0 or 1, where{' '}
              <InlineMath math="|0.5 - p|" /> is large) and{' '}
              <strong>expensive to buy low-consensus conjectures</strong> (price
              near 0.5, where <InlineMath math="|0.5 - p|" /> is near zero).
              The intuition: when the market is uncertain, your position is
              making a strong claim against a backdrop of genuine disagreement,
              and you should have to stake more to back it up. When the market
              has already converged, adding your voice to the consensus is
              low-stakes.
            </p>
            <p className="leading-relaxed mb-4">
              This has several appealing properties:
            </p>
            <ul className="space-y-4">
              <li className="leading-relaxed">
                <strong>It naturally discourages the trivially-true portfolio
                problem.</strong> Buying settled conjectures is cheap, but the
                returns are also near-zero (the price can&rsquo;t move much).
                The low cost matches the low information value. Meanwhile,
                taking a position on a contested conjecture at 0.50 is
                expensive, but the potential information gain is maximal. The
                cost curve tracks the bit potential curve.
              </li>
              <li className="leading-relaxed">
                <strong>It makes frontier conjectures expensive for a
                reason.</strong> When the market genuinely doesn&rsquo;t know
                whether something is true, claiming to know costs more. This
                is a form of epistemic humility enforcement: the market is
                saying &ldquo;if you want to move the price on something
                everyone is uncertain about, you&rsquo;d better mean it.&rdquo;
              </li>
              <li className="leading-relaxed">
                <strong>It preserves the anchoring function of settled
                conjectures.</strong> Since buying consensus conjectures is
                cheap, participants can freely express dependency on settled
                knowledge (&ldquo;water boils at 100&deg;C&rdquo;) without
                being penalized. The knowledge graph&rsquo;s bedrock remains
                accessible.
              </li>
            </ul>
            <p className="leading-relaxed mt-4 mb-4">
              But it also raises problems:
            </p>
            <ul className="space-y-4">
              <li className="leading-relaxed">
                <strong>It penalizes early movers on contrarian claims.</strong>{' '}
                Barry Marshall buying &ldquo;H. pylori causes ulcers&rdquo; at
                0.10 would face a relatively cheap position (high consensus
                that it&rsquo;s false, so{' '}
                <InlineMath math="|0.5 - 0.10| = 0.40" />, moderately cheap).
                But as his evidence pushes the price toward 0.50 &mdash; the
                zone of maximum uncertainty &mdash; the cost of additional
                shares spikes, making it expensive to keep pressing his case at
                exactly the moment when the market is starting to take him
                seriously.
              </li>
              <li className="leading-relaxed">
                <strong>It could suppress legitimate dissent.</strong> If you
                believe a conjecture at 0.50 should be at 0.90, the cost of
                pushing it there is maximized at the starting point. This might
                discourage participants from engaging with the most contested
                questions &mdash; precisely the questions where the market most
                needs informed participation.
              </li>
              <li className="leading-relaxed">
                <strong>The mapping is not obviously correct.</strong> Why{' '}
                <InlineMath math="|0.5 - p|" /> and not entropy{' '}
                <InlineMath math="H(p)" /> directly? Or the log-odds distance
                from 0.5? The right cost function depends on what behavior you
                want to incentivize, and different functions produce different
                market dynamics.
              </li>
            </ul>
            <p className="leading-relaxed mt-4">
              The deeper question is whether the cost of participation should be
              a function of the market state at all, or whether it should be
              fixed and let the returns do the work. In the standard LMSR, the
              cost is fixed per share but the expected return varies with price
              &mdash; buying at 0.50 has higher expected variance (could go
              either way) while buying at 0.95 has low expected return (not much
              room to move). The uncertainty-weighted approach moves this dynamic
              from the return side to the cost side. Whether that&rsquo;s better
              depends on which failure mode you&rsquo;re more worried about:
              trivially-true portfolio accumulation (fixed cost fails) or
              suppressed dissent on frontier questions (uncertainty-weighted cost
              fails).
            </p>
          </section>

        </div>
      </div>
    </div>
  )
}
