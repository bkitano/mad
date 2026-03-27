import { Link } from 'react-router-dom'
import { InlineMath } from 'react-katex'

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
            The behaviors we want the market to produce, stated explicitly.
            The pricing mechanism, scoring system, and market structure should
            be derived from these &mdash; not the other way around.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Why start here
          </h2>

          <p className="leading-relaxed">
            Most market design starts with a mechanism &mdash; an automated
            market maker, a scoring rule, a share supply model &mdash; and
            then asks what behaviors it produces. This is backwards. We should
            start with the behaviors we want and work backward to the
            mechanism that produces them. If we can&rsquo;t clearly state what
            the market should reward and punish, we can&rsquo;t evaluate
            whether any particular pricing function is correct.
          </p>

          <p className="leading-relaxed">
            The question &ldquo;what should the price be?&rdquo; is downstream
            of &ldquo;what is the price <em>for</em>?&rdquo;
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Does price = credence?
          </h2>

          <p className="leading-relaxed">
            In prediction markets, the price of a contract is typically
            interpreted as the market&rsquo;s credence that the event will
            occur. A price of 0.70 means &ldquo;the market thinks there is a
            70% chance.&rdquo; The conjecture market has inherited this
            assumption so far. But it is worth questioning whether price and
            credence should be the same number.
          </p>

          <p className="leading-relaxed">
            The price is the output of a mechanism &mdash; an AMM, an
            aggregation function, a market clearing process. The credence is
            a statement about the community&rsquo;s belief. These are only
            the same thing if the mechanism is calibrated so that its output
            can be read as a probability. This is a strong requirement. It
            means the mechanism must be incentive-compatible (participants
            are rewarded for honest revelation), well-calibrated (a price of
            0.70 implies the conjecture is true roughly 70% of the time across
            many conjectures at that price), and not systematically biased by
            the cost structure, the participant pool, or the share supply.
          </p>

          <p className="leading-relaxed">
            There is a case for separating them. The market could maintain
            two distinct numbers for each conjecture:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Price:</strong> the cost of a share, determined by the
              market mechanism. This is what participants pay and what
              determines returns. It need not be between 0 and 1. It could
              reflect supply and demand, information density, conjecture
              quality, or some combination.
            </li>
            <li className="leading-relaxed">
              <strong>Credence:</strong> the community&rsquo;s aggregate
              belief, derived from participants&rsquo; positions but not
              necessarily identical to the price. This is the number between
              0 and 1 that communicates &ldquo;how likely is this to be
              true.&rdquo; It could be a weighted average of participant
              beliefs, where weights come from stake size, track record, or
              both.
            </li>
          </ul>

          <p className="leading-relaxed">
            Under this separation, the price could account for things that
            credence should not: the information value of the conjecture, the
            cost of participation, the liquidity of the market. And the
            credence could account for things the price should not: the
            calibration history of the participants who hold positions, the
            quality of the evidence submitted, the structure of the
            attribution graph.
          </p>

          <p className="leading-relaxed">
            This is an open design question. Collapsing price and credence
            into a single number is elegant and legible. Separating them adds
            complexity but may produce better behavior. The rest of this page
            discusses the functions of &ldquo;price&rdquo; in the general
            sense &mdash; the cost and reward of participation &mdash; without
            assuming it must also be the credence readout.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            What is price for?
          </h2>

          <p className="leading-relaxed">
            Price in the conjecture market serves at least three distinct
            functions, and they may be in tension with each other:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>Signal.</strong> The price communicates the
              community&rsquo;s current credence. A price of 0.70 tells
              anyone who looks at it: &ldquo;the people who have thought about
              this believe there is roughly a 70% chance it is true.&rdquo;
              For this function, accuracy matters more than anything else.
            </li>
            <li className="leading-relaxed">
              <strong>Incentive.</strong> The price determines the returns
              from holding a position. If you buy at 0.40 and the price rises
              to 0.80, you have been rewarded for being right early. For this
              function, the spread between entry price and eventual price
              determines who gets rewarded and by how much.
            </li>
            <li className="leading-relaxed">
              <strong>Gate.</strong> The cost of taking a position determines
              who participates and how much influence they have. If positions
              are free, everyone piles in and the price is a popularity
              contest. If positions are expensive, only committed participants
              shape the price. For this function, cost determines the quality
              of the signal.
            </li>
          </ul>

          <p className="leading-relaxed">
            These three functions pull in different directions. A pure signal
            wants the price to move as freely as possible in response to
            information. A pure incentive wants the price to reward
            contrarians who are right. A pure gate wants the price to be
            expensive enough to exclude noise. The design challenge is finding
            a mechanism where all three functions reinforce each other rather
            than conflict.
          </p>

          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Desired behaviors
          </h2>

          <p className="leading-relaxed">
            The following are behaviors we want the market to produce. Each is
            stated as a principle, followed by what it implies about pricing.
          </p>

          {/* --- Behavior 1 --- */}
          <div
            className="rounded-lg border p-6 mt-6"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              1. Agreeing with consensus should be cheap
            </h3>
            <p className="leading-relaxed mb-3">
              If a conjecture is at 0.99 and you also believe it is true, you
              are not adding new information. You are joining a crowd. The
              market should make this easy and inexpensive, because settled
              knowledge forms the bedrock of the knowledge graph &mdash; many
              downstream conjectures depend on it, and participants should be
              able to declare those dependencies without friction.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>Implication for pricing:</strong> The cost of a position
              should decrease as the price approaches 0 or 1. When credence
              is near-certain in either direction, adding your voice to the
              consensus is low-cost. This is consistent with entropy-based
              pricing (<InlineMath math="H(p) \to 0" /> as{' '}
              <InlineMath math="p \to 0" /> or{' '}
              <InlineMath math="p \to 1" />) and inconsistent with{' '}
              <InlineMath math="|0.5 - p|" /> pricing (which makes consensus
              positions the <em>most</em> expensive).
            </p>
            <p className="leading-relaxed">
              <strong>But note the tension:</strong> if consensus is free, what
              prevents the{' '}
              <Link
                to="/platform/open-problems"
                className="underline"
                style={{ color: 'var(--accent)' }}
              >
                trivially-true portfolio problem
              </Link>
              ? Accumulating thousands of settled conjectures at zero cost
              would produce an enormous portfolio with no information content.
              The resolution may be that the <em>cost</em> is low but the{' '}
              <em>reward</em> is also low &mdash; scoring must discount
              positions in low-entropy conjectures so that cheap consensus
              positions don&rsquo;t generate unearned credit.
            </p>
          </div>

          {/* --- Behavior 2 --- */}
          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              2. Disagreeing with consensus should be expensive but high-reward
            </h3>
            <p className="leading-relaxed mb-3">
              If a conjecture is at 0.10 and you believe it should be at 0.80,
              you are making a strong claim that the community is wrong. The
              market should make you pay for this claim &mdash; not to punish
              you, but because the claim is extraordinary and the market needs
              to distinguish genuine insight from noise. If you are right, the
              reward should be proportionally large.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>Implication for pricing:</strong> The cost of moving the
              price away from consensus should scale with the magnitude of the
              move. This is already a property of LMSR: pushing the price from
              0.10 to 0.80 costs far more than nudging it from 0.10 to 0.12.
              The question is whether the cost should also depend on the
              current entropy (making it additionally expensive to move a
              contested price) or only on the distance of the move.
            </p>
            <p className="leading-relaxed">
              <strong>Historical test:</strong> Barry Marshall buying
              &ldquo;H. pylori causes ulcers&rdquo; at 0.10. The cost should
              be substantial enough that Marshall is putting something real on
              the line, but not so large that a researcher without institutional
              backing can&rsquo;t participate. The reward when the price later
              reaches 0.90 should be life-changing, because the information he
              contributed was life-changing.
            </p>
          </div>

          {/* --- Behavior 3 --- */}
          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              3. The market should attract participation where uncertainty is highest
            </h3>
            <p className="leading-relaxed mb-3">
              The conjectures the market most needs help with are the ones
              where credence is genuinely uncertain &mdash; price around 0.50,
              few participants, high entropy. The market should make it
              attractive to engage with these conjectures, not prohibitively
              expensive.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>Implication for pricing:</strong> This creates a direct
              tension with entropy-based cost. If{' '}
              <InlineMath math="H(p)" /> is the cost, then maximum-uncertainty
              conjectures are the most expensive to trade &mdash; exactly the
              opposite of what this principle demands. The resolution may be
              to separate cost from reward: make the cost moderate everywhere,
              but make the <em>scoring credit</em> proportional to entropy.
              You get more credit for participating in uncertain conjectures,
              not because it costs more, but because the market values the
              information more.
            </p>
            <p className="leading-relaxed">
              <strong>The attention allocation problem:</strong> If the market
              succeeds at this, it functions as an attention-allocation
              mechanism for the scientific community. The conjectures with the
              most to learn are the ones that draw the most participation.
              This is the opposite of how academic incentives currently work,
              where researchers are rewarded for publishing in established
              areas with clear methodologies, not for wading into maximally
              uncertain territory.
            </p>
          </div>

          {/* --- Behavior 4 --- */}
          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              4. Noise should be expensive
            </h3>
            <p className="leading-relaxed mb-3">
              Taking a position without genuine information should cost
              something. If the market allows unlimited free positions, it
              will be flooded with low-quality opinions that degrade the
              signal. The cost of a position is the market&rsquo;s defense
              against noise.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>Implication for pricing:</strong> There must be some
              nonzero cost to every position. The &ldquo;one free share for
              everyone&rdquo; model (see{' '}
              <Link
                to="/platform/open-problems"
                className="underline"
                style={{ color: 'var(--accent)' }}
              >
                Open Problems
              </Link>
              ) satisfies Behavior 1 (agreeing is cheap) but violates this
              one unless the free share has limited price influence. Perhaps
              the first share is free but has a weight of 1, while
              additional shares cost more but also carry more weight. The
              free tier gives everyone a voice; the paid tier determines who
              shapes the price.
            </p>
            <p className="leading-relaxed">
              <strong>Tension with Behavior 3:</strong> If noise is expensive,
              genuine participation in uncertain conjectures is also expensive,
              because the market can&rsquo;t distinguish signal from noise in
              advance. The cost either blocks both or permits both. The only
              way to resolve this is through <em>ex post</em> rewards:
              everyone pays the same cost, but participants who turn out to
              be right earn more back. The cost is a deposit, not a fee.
            </p>
          </div>

          {/* --- Behavior 5 --- */}
          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              5. Early correct participation should be rewarded more than late correct participation
            </h3>
            <p className="leading-relaxed mb-3">
              Someone who buys at 0.10 when the price later reaches 0.90 has
              contributed more to the market than someone who buys at 0.85
              after most of the evidence is already in. The early buyer took a
              risk, staked a contrarian claim, and was vindicated. The late
              buyer joined the consensus after it formed.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>Implication for pricing:</strong> This is naturally
              satisfied by any mechanism where your return is proportional to
              price movement after your entry. But it also implies that the
              cost of early participation should not be so high that it
              discourages contrarians. If the cost of buying at 0.10 is
              enormous (because you&rsquo;re far from consensus), only
              well-funded participants can be early. This concentrates the
              rewards of being right among those who can afford to be right.
            </p>
            <p className="leading-relaxed">
              <strong>The access problem:</strong> In the history of science,
              the people who are right early are often not the people with the
              most resources. Marshall was a junior researcher in Perth.
              Semmelweis was a Hungarian obstetrician dismissed by the Viennese
              establishment. Darwin worked independently. The market should
              not reproduce the power structures that the existing scientific
              establishment already has. If being right early is expensive,
              the market is just academia with extra steps.
            </p>
          </div>

          {/* --- Behavior 6 --- */}
          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              6. Generative conjectures should be rewarded more than terminal ones
            </h3>
            <p className="leading-relaxed mb-3">
              A conjecture that spawns ten downstream sub-conjectures, each
              generating its own evidence and trading activity, is more
              valuable to the knowledge graph than a conjecture that resolves
              in isolation. The market should recognize and reward this
              generativity.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>Implication for pricing:</strong> This is less about
              the cost of a position and more about the reward structure.
              Upstream royalties (see{' '}
              <Link
                to="/platform/scoring-metrics"
                className="underline"
                style={{ color: 'var(--accent)' }}
              >
                Scoring &amp; Metrics
              </Link>
              ) already address this: holders of parent conjectures earn
              residuals from downstream activity. But the pricing mechanism
              could reinforce it: perhaps the cost of a position in a
              conjecture with many active descendants should be higher
              (reflecting its proven value) while the cost of a new,
              unconnected conjecture should be lower (encouraging
              exploration).
            </p>
            <p className="leading-relaxed">
              <strong>Risk:</strong> If the market over-rewards generativity,
              participants will game it by creating artificial sub-conjecture
              trees &mdash; splitting one claim into many fragments to
              inflate downstream counts. The market needs to distinguish
              genuine generativity (sub-conjectures that are independently
              testable and trade on their own merits) from artificial
              fragmentation (splitting a single claim into pieces that
              aren&rsquo;t independently meaningful).
            </p>
          </div>

          {/* --- Behavior 7 --- */}
          <div
            className="rounded-lg border p-6 mt-4"
            style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-widest mb-3"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
            >
              7. The price should be hard to manipulate
            </h3>
            <p className="leading-relaxed mb-3">
              No single participant or coordinated group should be able to
              move the price significantly without committing proportional
              resources. The cost of manipulation should scale with the size
              of the desired price movement.
            </p>
            <p className="leading-relaxed mb-3">
              <strong>Implication for pricing:</strong> This is a core
              property of LMSR and most AMMs: the cost of moving the price
              from <InlineMath math="p_1" /> to <InlineMath math="p_2" /> is
              a function of the distance, mediated by the liquidity
              parameter. Large moves require large stakes. This is
              non-negotiable &mdash; any mechanism that allows cheap large
              price moves is vulnerable to manipulation.
            </p>
            <p className="leading-relaxed">
              <strong>Tension with Behavior 5:</strong> If large moves are
              expensive, then contrarians who are right must pay a lot to
              express their view. The Marshall who wants to push the price
              from 0.10 to 0.80 faces the same cost as the manipulator who
              wants to push the price from 0.50 to 0.90 for strategic
              reasons. The market can&rsquo;t distinguish them in advance. The
              only resolution is ex post: both pay the same cost, but the one
              who was right earns it back (and more), while the one who was
              wrong loses it.
            </p>
          </div>

          {/* --- Summary --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            The tensions
          </h2>

          <p className="leading-relaxed">
            Several of these behaviors are in direct conflict:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              <strong>1 vs. 4:</strong> Consensus should be cheap, but noise
              should be expensive. If consensus positions are free, noise
              participants can accumulate them costlessly.
            </li>
            <li className="leading-relaxed">
              <strong>3 vs. 4:</strong> The market should attract
              participation at maximum uncertainty, but noise should be
              expensive. At maximum uncertainty, the market can&rsquo;t tell
              signal from noise.
            </li>
            <li className="leading-relaxed">
              <strong>5 vs. 7:</strong> Early contrarians should be rewarded,
              but the price should be hard to manipulate. Both early
              contrarians and manipulators want to make large price moves.
            </li>
          </ul>

          <p className="leading-relaxed">
            The common resolution across all three tensions is the same
            pattern: <strong>separate the cost of entry from the reward for
            being right.</strong> The cost is a deposit that makes noise and
            manipulation expensive. The reward is an ex post return that
            pays back contrarians who were correct and penalizes those who
            were wrong. Under this model:
          </p>

          <ul className="space-y-3 ml-6 list-disc">
            <li className="leading-relaxed">
              Consensus positions cost little (Behavior 1) and earn little
              (no reward for agreeing with what everyone already knows).
            </li>
            <li className="leading-relaxed">
              Frontier positions cost a moderate deposit (Behavior 4) and
              earn large returns if correct (Behaviors 2, 3, 5) or forfeit
              the deposit if wrong (Behavior 7).
            </li>
            <li className="leading-relaxed">
              The scoring system, not the pricing mechanism, distinguishes
              information from noise after the fact (Behavior 6).
            </li>
          </ul>

          <p className="leading-relaxed">
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
