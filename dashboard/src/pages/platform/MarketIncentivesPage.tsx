import { Link } from 'react-router-dom'
import { InlineMath, BlockMath } from 'react-katex'

export default function MarketIncentivesPage() {
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
            Market Incentives
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            How the market should behave, what it should reflect, and how
            we get there.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* ============================================================
              SECTION 0: FIRST PRINCIPLES
              ============================================================ */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            First principles
          </h2>

          <p className="leading-relaxed">
            What is price? In a traditional market, the price of a share is
            a single number that simultaneously serves multiple roles. In the
            conjecture market as it stands, price does at least three things
            at once:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Reward.</strong> The price determines returns. If you
              buy at 0.20 and the price rises to 0.80, the difference is your
              payoff. Price is the mechanism that compensates participants for
              being right.
            </li>
            <li className="leading-relaxed">
              <strong>Barrier to entry.</strong> The price is what you pay to
              take a position. It determines who participates and how much
              influence they have. A high price filters out casual opinion; a
              low price invites it in.
            </li>
            <li className="leading-relaxed">
              <strong>Estimator of consensus.</strong> The price is read as the
              community&rsquo;s credence. A price of 0.70 is interpreted as
              &ldquo;the market thinks there is a 70% chance this is
              true.&rdquo; The price is the public signal.
            </li>
          </ul>

          <p className="leading-relaxed">
            The question is: <em>should these be the same number?</em>
          </p>

          <p className="leading-relaxed">
            In traditional prediction markets, collapsing all three into a
            single price is elegant and legible. But it also means that every
            design decision about one function constrains the others. Making
            the barrier to entry low (to encourage participation) changes the
            reward structure. Optimizing the price as a credence estimator
            (calibration, accuracy) may conflict with its role as an incentive
            mechanism.
          </p>

          <p className="leading-relaxed">
            An automated market does not have to couple these tightly. We
            could maintain separate numbers: a <em>credence</em> that is
            the public signal, a <em>cost</em> that determines the barrier
            to entry, and a <em>reward</em> that determines payoffs &mdash;
            each governed by its own logic, tuned to its own purpose.
          </p>

          <p className="leading-relaxed">
            In fact, for the reward function, we don&rsquo;t really care
            about a portfolio&rsquo;s resting credence value at all. A
            participant who holds a conjecture at 0.90 when everyone else
            also holds it at 0.90 has not demonstrated insight. What we care
            about is whether the portfolio was <em>right before others knew
            it</em> &mdash; whether the participant took a position when the
            credence was low and was vindicated by later evidence and
            community convergence. The reward should track the <em>delta</em>{' '}
            between when you believed it and when the community caught up,
            not the final resting price.
          </p>

          <p className="leading-relaxed">
            This further argues for decoupling: the credence is a snapshot of
            where the community is <em>now</em>. The reward is a function of
            how far ahead of the community you were, and for how long. These
            are fundamentally different quantities. The rest of this page
            explores what each should look like.
          </p>

          {/* ============================================================
              SECTION 1: WHAT WE WANT PARTICIPANTS TO DO
              ============================================================ */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            What we want participants to do
          </h2>

          <p className="leading-relaxed">
            The market exists to produce a few specific behaviors from its
            participants. Everything else &mdash; the mechanism, the pricing
            function, the scoring rules &mdash; is downstream of this list.
          </p>

          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              1. Buy conjectures early
            </h3>
            <p className="leading-relaxed mb-3">
              If you have genuine insight that a conjecture is true (or false)
              before the rest of the community does, you should buy shares.
              The market should reward you for being right early and taking a
              risk when others hadn&rsquo;t yet recognized the signal. Someone
              who buys at 0.10 when the price later reaches 0.90 has
              contributed more than someone who buys at 0.85 after the
              evidence is already public.
            </p>
            <p className="leading-relaxed">
              <strong>The access problem:</strong> In the history of science,
              the people who are right early are often not the people with
              resources. Marshall was a junior researcher in Perth.
              Semmelweis was dismissed by the Viennese establishment. The
              market should not reproduce existing power structures. If being
              right early is expensive, the market is just academia with extra
              steps.
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
              2. Publish evidence that updates posteriors
            </h3>
            <p className="leading-relaxed mb-3">
              Participants should be rewarded for submitting evidence &mdash;
              experimental results, theoretical arguments, data &mdash; that
              causes Bayesian updates in the community&rsquo;s beliefs. The
              market should not just reward opinion; it should reward the
              production of <em>reasons</em> for changing opinions. Publishing
              a strong result that moves the credence of a conjecture from
              0.40 to 0.70 is the kind of contribution the market most wants
              to incentivize.
            </p>
            <p className="leading-relaxed">
              The natural workflow: you have research showing B is likely true.
              You buy B shares. Then you publish. The publication drives the
              price up. You are rewarded both for being right and for making
              the community smarter.
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
              3. Build the dependency graph through trades
            </h3>
            <p className="leading-relaxed mb-3">
              If you believe B depends on A, your trading behavior should
              reflect this. You buy A and B together, then publish evidence
              for B. The publication drives B up, and A should also rise in
              proportion to how much the community believes B depends on A.
              This creates an implicit Bayesian network in the trade graph
              &mdash; not one that anyone declared explicitly, but one that
              emerges from the pattern of correlated positions and evidence
              submission.
            </p>
            <p className="leading-relaxed">
              The trades themselves encode the dependency structure. If many
              participants hold both A and B, and evidence for B consistently
              moves A&rsquo;s price, the market has learned the conditional
              relationship <InlineMath math="P(A \mid B)" /> without anyone
              having to define it.
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
              4. Engage where uncertainty is highest
            </h3>
            <p className="leading-relaxed mb-3">
              The conjectures the market most needs help with are the ones
              where credence is genuinely uncertain &mdash; price around 0.50,
              few participants, high entropy. The market should make it
              attractive to engage with these conjectures.
            </p>
            <p className="leading-relaxed">
              If the market succeeds at this, it functions as an
              attention-allocation mechanism for the scientific community. The
              conjectures with the most to learn draw the most participation.
              This is the opposite of how academic incentives currently work,
              where researchers are rewarded for publishing in established
              areas, not for wading into maximally uncertain territory.
            </p>
          </div>

          {/* ============================================================
              SECTION 2: WHAT WE WANT THE MARKET TO REFLECT
              ============================================================ */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            What we want the market to reflect
          </h2>

          <p className="leading-relaxed">
            We want the market to produce at least two distinct numbers for
            each conjecture, and they are not the same thing:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Credence:</strong> a number between 0 and 1 reflecting
              the community&rsquo;s aggregate belief in whether the conjecture
              is true. This number should be <em>publicly visible</em>. Anyone
              should be able to look at a conjecture and see what the
              community thinks. This is the market&rsquo;s signal to the world.
            </li>
            <li className="leading-relaxed">
              <strong>Reward:</strong> a number reflecting how much a
              participant has been compensated for good scientific intuition.
              You should be rewarded for buying conjectures early, for
              publishing evidence that updates posteriors, and for building
              positions in dependency chains that turn out to be correct. An
              increase in portfolio value should track good scientific
              intuition, not just good trading.
            </li>
          </ul>

          <p className="leading-relaxed">
            The credence is the market&rsquo;s public face. The reward is the
            market&rsquo;s private incentive mechanism. They are related
            &mdash; rewards flow from changes in credence &mdash; but they are
            not the same number and should not be conflated.
          </p>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            How LMSR maps shares to credence
          </h3>

          <p className="leading-relaxed">
            The core mechanism we build on is the Logarithmic Market Scoring
            Rule (LMSR). The key equation is simple:
          </p>

          <BlockMath math="P_i = \frac{e^{Q_i / b}}{\sum_j e^{Q_j / b}}" />

          <p className="leading-relaxed">
            where <InlineMath math="P" /> is the vector of <em>credences</em>{' '}
            across all positions (e.g., YES and NO),{' '}
            <InlineMath math="Q" /> is the vector of outstanding share
            quantities, and <InlineMath math="b" /> is a liquidity parameter
            that controls how sensitive credences are to trades. Crucially,{' '}
            <InlineMath math="P" /> here is <em>not</em> the price you pay
            for a share &mdash; as we argued above, price and credence need
            not be the same number. The softmax gives us a credence
            estimator:
          </p>

          <BlockMath math="P = \text{softmax}(Q / b)" />

          <p className="leading-relaxed">
            The principle is this: we need a function that maps a vector of
            share quantities to a <em>probability simplex</em> &mdash; a
            vector whose components are non-negative and sum to 1. The
            softmax function does this. When someone buys YES shares,{' '}
            <InlineMath math="Q_{\text{YES}}" /> increases, and the softmax
            pushes <InlineMath math="P_{\text{YES}}" /> up and{' '}
            <InlineMath math="P_{\text{NO}}" /> down, maintaining the
            constraint <InlineMath math="P_{\text{YES}} + P_{\text{NO}} = 1" />.
          </p>

          <p className="leading-relaxed">
            The liquidity parameter <InlineMath math="b" /> controls the
            tradeoff between sensitivity and stability. Small{' '}
            <InlineMath math="b" /> means prices react sharply to each
            trade &mdash; good for thin markets where every participant
            matters, but vulnerable to manipulation. Large{' '}
            <InlineMath math="b" /> means prices are sluggish &mdash; hard
            to manipulate, but you need many participants to move the price
            to where it should be.
          </p>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Why softmax? Why not something else?
          </h3>

          <p className="leading-relaxed">
            There are many functions that map a vector of quantities to a
            probability simplex. The softmax (with its exponentials) is the
            canonical choice, but it is not the only one. For example, a
            simple <strong>power normalization</strong>:
          </p>

          <BlockMath math="P_i = \frac{Q_i^{\alpha}}{\sum_j Q_j^{\alpha}}" />

          <p className="leading-relaxed">
            also maps positive share quantities to a probability simplex.
            The choice of normalization function determines the <em>shape</em>{' '}
            of how prices respond to trades. The exponential in softmax means
            that buying shares has a <em>multiplicative</em> effect on the
            odds ratio &mdash; each additional share multiplies the odds by a
            constant factor. A power normalization would give a different
            price response curve: more linear for low{' '}
            <InlineMath math="\alpha" />, more winner-take-all for high{' '}
            <InlineMath math="\alpha" />.
          </p>

          <p className="leading-relaxed">
            The nature of the normalization function determines the market&rsquo;s
            character. LMSR&rsquo;s exponential normalization has nice
            information-theoretic properties &mdash; the cost function is the
            log-partition function, the prices are maximum-entropy given the
            trades, and there are bounded worst-case losses for the market
            maker. But whether those properties are the <em>right</em> ones
            for a conjecture market is an open question.
          </p>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            An uncertainty market, not a conjecture market
          </h3>

          <p className="leading-relaxed">
            If we take the decoupling argument seriously &mdash; credence is
            not price, and the thing we reward is being right before others
            &mdash; then we arrive at a reframing of what the market actually
            trades. It is not really a conjecture market. It is
            an <em>uncertainty market</em>.
          </p>

          <p className="leading-relaxed">
            Here is the invariant we want: things that people believe are
            true should have low value. The more people believe in a truth,
            the cheaper it should be to hold. This sounds paradoxical if you
            think of the market as trading conjectures, where &ldquo;the
            market believes X&rdquo; should make X valuable. But it makes
            perfect sense if you think of the market as trading <em>
            uncertainty</em>.
          </p>

          <p className="leading-relaxed">
            When a conjecture is first proposed, uncertainty is high. There
            is a lot of entropy &mdash; the community genuinely does not know
            whether it is true. This uncertainty is the scarce resource. As
            evidence comes in, uncertainty decreases. The credence moves
            toward 0 or 1, entropy drops, and the conjecture becomes
            &ldquo;settled.&rdquo; The uncertainty has been consumed by
            evidence.
          </p>

          <p className="leading-relaxed">
            The market, then, is a mechanism for pricing and allocating
            uncertainty. Early participants are buying uncertainty when it is
            abundant and unresolved. They are rewarded when that uncertainty
            is later resolved &mdash; when evidence collapses the entropy.
            Late participants are buying into a conjecture whose uncertainty
            has already been consumed; there is little left for them to
            contribute, and correspondingly little reward.
          </p>

          <p className="leading-relaxed">
            This framing clarifies the pricing question. The <em>cost</em>{' '}
            of a position should be related to the remaining uncertainty
            &mdash; something like the entropy{' '}
            <InlineMath math="H(P)" /> of the current credence distribution.
            Near the extremes (<InlineMath math="P \approx 0" /> or{' '}
            <InlineMath math="P \approx 1" />), entropy is low, cost is low,
            and reward is low. Near maximum uncertainty ({' '}
            <InlineMath math="P \approx 0.5" />), entropy is high, cost is
            higher, and the potential reward for resolving that uncertainty is
            large.
          </p>

          <p className="leading-relaxed">
            But entropy alone is not enough. Entropy prices the <em>cost</em>{' '}
            of participation &mdash; how much uncertainty you are buying into.
            It is symmetric: at <InlineMath math="P = 0.25" /> and{' '}
            <InlineMath math="P = 0.75" />, the entropy is the same, and
            so is the cost of taking a position. But the <em>reward</em>{' '}
            must be directional. Two participants can buy into the same
            entropy &mdash; one buying YES, one buying NO &mdash; and only
            one of them should be rewarded, namely the one whose position
            the consensus eventually moves toward.
          </p>

          <p className="leading-relaxed">
            This gives us a clean separation. The cost function is
            entropy: you pay for uncertainty regardless of which side you
            take. The reward function is directional: you earn in proportion
            to how much entropy you bought <em>times</em> how far the
            consensus moved toward your position. Buying at high entropy in
            the direction the world eventually goes is the most rewarded
            action. Buying at high entropy in the wrong direction is the most
            penalized. Buying at low entropy in either direction is cheap and
            earns little, because there was little uncertainty left to resolve.
          </p>

          <p className="leading-relaxed">
            What is being traded is not belief. It is the <em>resolution of
            doubt</em> &mdash; and the direction matters.
          </p>

          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Historical example: H. pylori and ulcers
            </h3>
            <p className="leading-relaxed mb-3">
              In 1982, Barry Marshall and Robin Warren proposed that peptic
              ulcers were caused by the bacterium <em>Helicobacter pylori</em>,
              not by stress or diet. Suppose the conjecture &ldquo;<em>H. pylori</em>{' '}
              causes peptic ulcers&rdquo; had been on the market.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>1982 &mdash; conjecture proposed.</strong> No one owns
              any shares. Credence: <InlineMath math="P = 0.50" /> by default.
              Entropy is at maximum:{' '}
              <InlineMath math="H = 1.0" /> bit. The cost of a position is
              high, because the uncertainty is genuinely unresolved and the
              market has no information. Marshall buys YES. He is buying{' '}
              <em>peak uncertainty</em>. But note: someone else might look at
              this conjecture and buy NO, equally convinced the establishment
              is right. Both pay the same high cost. The entropy prices
              uncertainty itself, not direction.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>1983&ndash;1984 &mdash; early evidence.</strong> Marshall
              cultures the bacterium, then famously drinks a petri dish of it
              and develops gastritis. A handful of researchers take notice,
              but most of the establishment doubles down on the stress
              hypothesis. Credence shifts to{' '}
              <InlineMath math="P \approx 0.25" /> &mdash; more people are
              actively buying NO than YES. Entropy drops slightly:{' '}
              <InlineMath math="H \approx 0.81" /> bits. The cost of a
              position has come down, but only slightly &mdash; there is
              still plenty of unresolved uncertainty.
            </p>
            <p className="leading-relaxed mb-3">
              Here is where directionality matters. The NO buyers at{' '}
              <InlineMath math="P = 0.25" /> paid into real uncertainty, but
              they bought in the direction the consensus would eventually move{' '}
              <em>away from</em>. Their shares should lose value as the
              credence later swings toward YES. The reward is not just
              &ldquo;you bought at high entropy&rdquo; &mdash; it is
              &ldquo;you bought at high entropy <em>and</em> the consensus
              moved toward your position.&rdquo; Buying uncertainty in the
              wrong direction is a losing trade.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>1990s &mdash; accumulating trials.</strong> Randomized
              controlled trials show that antibiotic treatment cures ulcers.
              The evidence is hard to deny. Credence climbs:{' '}
              <InlineMath math="P \approx 0.75" />. Entropy:{' '}
              <InlineMath math="H \approx 0.81" /> bits. New participants can
              still buy YES, but the entropy is symmetric around 0.50
              &mdash; a credence of 0.75 has the same entropy as 0.25. The
              difference is that Marshall&rsquo;s YES shares, bought when the
              consensus was against him, have been gaining value the entire
              way. A new YES buyer at 0.75 is joining an emerging consensus;
              their potential upside is the remaining distance
              to <InlineMath math="P = 1" />, not the full journey from 0.50.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>2005 &mdash; Nobel Prize.</strong> Marshall and Warren
              receive the Nobel Prize in Physiology or Medicine. Credence:{' '}
              <InlineMath math="P \approx 0.99" />. Entropy:{' '}
              <InlineMath math="H \approx 0.08" /> bits. The conjecture is
              settled. A YES position now costs almost nothing and earns
              almost nothing &mdash; there is no uncertainty left to resolve.
              The NO holders from 1984 have long since lost their stakes.
            </p>
            <p className="leading-relaxed">
              The lesson: entropy determines the <em>cost</em> of participation
              &mdash; how much uncertainty you are buying into. But the{' '}
              <em>reward</em> is directional: you are rewarded in proportion to
              how much entropy you bought <em>times</em> how far the consensus
              moved toward your position. Marshall at{' '}
              <InlineMath math="H = 1.0" /> buying YES earned the most,
              because he bore maximum uncertainty in the direction that
              turned out to be right. The establishment doctors who bought NO
              at the same entropy earned nothing &mdash; they bore the same
              uncertainty but the world moved the other way.
            </p>
          </div>

          <h3
            className="text-xl font-bold mt-8"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Implicit Bayesian networks
          </h3>

          <p className="leading-relaxed">
            Some of the numbers the market produces should not be visible as
            explicit fields on a conjecture, but should emerge from the
            structure of trading. When participants buy correlated positions
            across conjectures &mdash; A and B together, because they believe
            B depends on A &mdash; the trade graph encodes conditional
            dependencies. If evidence for B consistently moves A&rsquo;s
            price, the market has implicitly learned{' '}
            <InlineMath math="P(A \mid B)" />.
          </p>

          <p className="leading-relaxed">
            This is a form of decentralized structure learning. No one
            declares the Bayesian network. It emerges from the overlap of
            portfolios and the propagation of evidence through prices.
            Whether this actually works &mdash; whether the implicit
            conditional dependencies are well-calibrated &mdash; depends on
            the details of the pricing mechanism and how evidence propagation
            is handled.
          </p>

          {/* ============================================================
              SECTION 3: HOW WE DO IT
              ============================================================ */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            How we do it
          </h2>

          <p className="leading-relaxed">
            Producing the behaviors above while reflecting the right numbers
            requires navigating several design tensions. The principles below
            are sometimes mutually exclusive &mdash; the mechanism design is
            a search for the best tradeoff surface.
          </p>

          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              Reward impactful work
            </h3>
            <p className="leading-relaxed mb-3">
              The market should disproportionately reward conjectures and
              evidence that have high impact. The open question is how to
              measure impact. One proxy: conjecture volume &mdash; a
              conjecture that spawns many downstream sub-conjectures, each
              generating its own evidence and trading activity, is more
              valuable to the knowledge graph than one that resolves in
              isolation. Upstream royalties (holders of parent conjectures
              earning residuals from downstream activity) are one mechanism
              for this.
            </p>
            <p className="leading-relaxed">
              <strong>Risk:</strong> If over-rewarded, participants will
              game generativity by creating artificial sub-conjecture trees.
              The market needs to distinguish genuine generativity (independently
              testable sub-conjectures) from artificial fragmentation.
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
              Be hard to abuse
            </h3>
            <p className="leading-relaxed mb-3">
              No single participant or coordinated group should be able to
              move the price significantly without committing proportional
              resources. This is a core property of LMSR: the cost of moving
              the price from <InlineMath math="p_1" /> to{' '}
              <InlineMath math="p_2" /> is a function of the distance,
              mediated by the liquidity parameter. Large moves require large
              stakes.
            </p>
            <p className="leading-relaxed">
              <strong>Tension:</strong> If large moves are expensive, then
              contrarians who are right face the same cost barrier as
              manipulators. The market can&rsquo;t distinguish them in
              advance. The resolution is ex post: both pay the same cost, but
              the one who was right earns it back (and more), while the wrong
              one loses it.
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
              Things everyone believes should have low barrier to ownership
            </h3>
            <p className="leading-relaxed mb-3">
              If a conjecture is at 0.99 and you also believe it is true, you
              are not adding new information. The market should make this easy
              and inexpensive, because settled knowledge forms the bedrock of
              the knowledge graph &mdash; many downstream conjectures depend
              on it. But the reward for late consensus should also be low.
              The people who should be rewarded are the ones who owned it when
              fewer people believed it.
            </p>
            <p className="leading-relaxed">
              This is consistent with entropy-based pricing:{' '}
              <InlineMath math="H(p) \to 0" /> as{' '}
              <InlineMath math="p \to 0" /> or{' '}
              <InlineMath math="p \to 1" />, so near-certain positions are
              cheap. But the <em>reward</em> must also be discounted at
              low entropy, or else participants accumulate thousands of
              settled conjectures at zero cost for unearned credit.
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
              Portfolio value should track scientific intuition
            </h3>
            <p className="leading-relaxed mb-3">
              An increase in portfolio value should mean the participant has
              demonstrated good scientific intuition &mdash; not just good
              trading instincts. This is the deepest constraint on the
              mechanism design. If the market can be profited by pure
              game-theoretic arbitrage with no domain knowledge, it has failed
              at its purpose.
            </p>
            <p className="leading-relaxed">
              The reward function must be calibrated so that the dominant
              strategy is: form genuine beliefs based on evidence, take
              positions reflecting those beliefs, produce evidence that moves
              the community&rsquo;s beliefs, and get rewarded for the movement.
              Pure speculation without evidence production should be a losing
              strategy in expectation.
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
              Separate cost from reward
            </h3>
            <p className="leading-relaxed mb-3">
              Many of the tensions above resolve with the same pattern:
              <strong> separate the cost of entry from the reward for being
              right.</strong> The cost is a deposit that makes noise and
              manipulation expensive. The reward is an ex post return that
              pays back participants who were correct.
            </p>
            <ul className="space-y-2 ml-6 list-disc">
              <li className="leading-relaxed">
                Consensus positions cost little and earn little (no reward
                for agreeing with what everyone already knows).
              </li>
              <li className="leading-relaxed">
                Frontier positions cost a moderate deposit and earn large
                returns if correct, or forfeit the deposit if wrong.
              </li>
              <li className="leading-relaxed">
                The scoring system, not the pricing mechanism, distinguishes
                information from noise after the fact.
              </li>
            </ul>
          </div>

          <p className="leading-relaxed mt-6">
            This suggests the pricing mechanism and the scoring mechanism are
            two halves of the same system. The price determines the cost of
            entry. The score determines the reward. Neither alone produces
            the behaviors we want. See{' '}
            <Link
              to="/platform/price-determination"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Price Determination
            </Link>
            {' '}and{' '}
            <Link
              to="/platform/scoring-metrics"
              className="underline"
              style={{ color: 'var(--accent)' }}
            >
              Scoring &amp; Metrics
            </Link>
            {' '}for the current state of each.
          </p>

        </div>
      </div>
    </div>
  )
}
