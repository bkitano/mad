import { Link } from 'react-router-dom'

interface Example {
  label: string
  statement: string
  critiques: string[]
  trades: string[]
  market: string
}

function ExampleCard({ example }: { example: Example }) {
  return (
    <div
      className="rounded-lg border p-6"
      style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
    >
      <h3
        className="text-sm font-semibold uppercase tracking-widest mb-3"
        style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
      >
        {example.label}
      </h3>
      <p
        className="leading-relaxed mb-4 text-lg"
        style={{ fontFamily: 'var(--font-body)', fontStyle: 'italic' }}
      >
        &ldquo;{example.statement}&rdquo;
      </p>
      <div className="space-y-3">
        <div>
          <h4
            className="text-xs font-semibold uppercase tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
          >
            Critiques
          </h4>
          <ul className="space-y-1">
            {example.critiques.map((c, i) => (
              <li key={i} className="leading-relaxed text-sm">{c}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4
            className="text-xs font-semibold uppercase tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
          >
            Example trades
          </h4>
          <ul className="space-y-1">
            {example.trades.map((t, i) => (
              <li key={i} className="leading-relaxed text-sm">{t}</li>
            ))}
          </ul>
        </div>
        <div>
          <h4
            className="text-xs font-semibold uppercase tracking-widest mb-2"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--accent)' }}
          >
            Market response
          </h4>
          <p className="leading-relaxed text-sm">{example.market}</p>
        </div>
      </div>
    </div>
  )
}

const examples: Example[] = [
  {
    label: 'Already consensus',
    statement:
      'Water boils at 100\u00B0C at standard atmospheric pressure (1 atm).',
    critiques: [
      'This is already universally known. You might think it has no market value — price is already at 1.0, no trading opportunity.',
    ],
    trades: [
      'Nearly every participant in the physical sciences holds a position here, because their own conjectures depend on basic thermodynamic facts. A chemical engineer working on distillation processes needs this to be true for their downstream conjectures about separation efficiency to be coherent. A climate scientist modeling ocean evaporation rates implicitly depends on it. The buy-side volume is massive — not because anyone is uncertain, but because holding a position is a way of declaring a dependency.',
      'A market maker uses it as a reference asset to calibrate spread algorithms — its known stability makes it useful as a benchmark, similar to how bond traders use Treasuries.',
      'Since there is no limit on the number of shares (anyone can buy in at any time), the volume is effectively unbounded. The interesting question is not whether people trade it, but what it costs to hold a position — and whether there should be any cost at all for stating what everyone already knows. See Open Problems for more on the economics of share supply.',
    ],
    market:
      'Price sits at ~0.99 with a razor-thin spread. Volume is massive but almost entirely one-directional — virtually everyone is on the buy side, because an enormous number of downstream conjectures depend on basic thermodynamic facts being true. Not every conjecture needs to be controversial to be valid. It serves as an anchor: if someone proposes a conjecture that implies water doesn\'t boil at 100\u00B0C, the market can immediately price that new conjecture down by referencing this one. Settled conjectures are the bedrock of the market\'s knowledge graph.',
  },
  {
    label: 'Provably false',
    statement:
      'Heavier objects fall faster than lighter objects in a vacuum.',
    critiques: [
      'This is straightforwardly wrong. Galileo demonstrated it in the 1600s. Every physics student can verify it.',
      'You might think this wastes the market\'s time — but it doesn\'t.',
    ],
    trades: [
      'A physics undergraduate takes a position against it, confident from introductory mechanics. But since the price is already at ~0.01, there\'s almost nothing to gain — the market has already priced in the falsity. There are no "easy returns" here; the information is already reflected.',
      'A contrarian philosopher attempts to buy, arguing about operational definitions of "vacuum" and "heavier." The market absorbs their position without meaningful price impact because the sell-side volume overwhelms them.',
      'As new participants join the market (new physics students every semester, new researchers entering adjacent fields), they each take a position against the conjecture. This produces uniformly high volume over time and a price that decreases monotonically — not because new evidence arrives, but because each new informed participant adds to the consensus.',
    ],
    market:
      'Price crashes to ~0.01 early and stays there. This is the market working correctly — the price signal broadcasts "this is wrong" to anyone watching. Volume is uniformly high because every new participant who encounters the conjecture recognizes it as false and takes a position against it. The price converges fast and then continues to decrease monotonically (approaching but never reaching 0.00) as the population of informed participants grows. The conjecture is bad science, but it\'s a perfectly fine market instrument — it just has trivial dynamics.',
  },
  {
    label: 'Low predictive capacity (doesn\'t generalize)',
    statement:
      'GPT-4 scores 86.4% on the MMLU benchmark.',
    critiques: [
      'This is a measurement, not a conjecture. It describes a single, already-observed data point.',
      'It generates no predictions about anything else. It tells you nothing about GPT-5, nothing about other benchmarks, nothing about capability in general.',
      'There\'s nothing to trade against because there\'s no uncertainty about the future.',
    ],
    trades: [
      'An ML benchmark tracker buys at 0.95 for a riskless fractional return — the equivalent of picking up pennies.',
      'No researcher has work that depends on this claim. No one trades it after initial convergence because there is literally nothing left to learn.',
    ],
    market:
      'Price immediately converges to ~1.0 (or ~0.0 if wrong). No trade volume after initial convergence because there\'s nothing left to learn. No downstream conjectures — it\'s a dead end. Identical to betting on a coin flip that already happened.',
  },
  {
    label: 'Trivially narrow',
    statement:
      'The next paper published by the AlphaFold team will have more than 12 authors.',
    critiques: [
      'Technically falsifiable — you can check the author list. But it generates zero scientific insight.',
      'Knowing the answer tells you nothing about protein folding, team dynamics in general, or anything else.',
      'It has no downstream conjectures. Nothing depends on it.',
    ],
    trades: [
      'A DeepMind insider could trade on private knowledge of the author list, but the position is so small it\'s not worth the reputational risk.',
      'A random speculator takes a flyer at 0.50. They\'re essentially gambling, and they know it.',
      'Everyone else ignores it. The order book is empty.',
    ],
    market:
      'Very low volume. No researcher has work that depends on this claim. Price sensitivity is binary — it resolves exactly once, and no intermediate evidence updates it meaningfully. The market treats it as noise. Even if you get it right, your portfolio learns nothing from it — it doesn\'t improve your veracity score on anything that matters.',
  },
  {
    label: 'Tautological / unfalsifiable',
    statement:
      'In efficient markets, all available information is reflected in asset prices.',
    critiques: [
      'If you define "efficient" as "all information is reflected in prices," this is a tautology — true by definition.',
      'Any apparent mispricing is explained away: either the market wasn\'t efficient in that instance, or you don\'t have the same information the market does.',
      'No possible observation can count as evidence against it, because the escape hatch is built into the definition.',
    ],
    trades: [
      'A philosophy of science researcher recognizes the tautology and avoids it entirely — there\'s no epistemically coherent position to take.',
      'A finance PhD attempts to buy at 0.70, interpreting the statement empirically rather than definitionally. But what does it even mean to take a position against this conjecture? In a market with unlimited share supply and no counterparty requirement, the question becomes: who is on the other side, and what would they be claiming? If shares are unlimited, there\'s no need for a counterparty to sell you the shares — but then the "market price" is just an aggregation of opinions with no friction, and it\'s unclear what economic signal it produces. (See Open Problems on counterparty mechanics.)',
      'A market quality analyst flags it using conjecture processability metrics: zero price sensitivity to evidence, undefined resolution criteria, no downstream conjectures. It scores in the bottom percentile on every quality dimension (see Scoring & Metrics).',
    ],
    market:
      'The market cannot process this conjecture. No evidence moves the price, so the price is arbitrary and stays wherever it was initially set. Zero trade volume — sophisticated participants recognize it as untradeable. Wide spread because the few participants who do trade are essentially guessing about a meaningless number.',
  },
  {
    label: 'Specific and falsifiable',
    statement:
      'Transformer models with fewer than 1B parameters can achieve >90% accuracy on MATH benchmark problems when trained with verifier-guided search.',
    critiques: [
      'Clear threshold (1B params, 90% accuracy), specific benchmark (MATH), specific method (verifier-guided search).',
      'Any team can attempt to reproduce it. Success or failure moves the price directly.',
    ],
    trades: [
      'An ML researcher at a lab working on small-model efficiency buys heavily at 0.40 based on unpublished internal results showing 85% accuracy at 800M parameters. They believe the last 5% is achievable with their new verifier architecture. This is analogous to how biotech insiders legally trade based on domain expertise that hasn\'t yet been published.',
      'A scaling-laws maximalist sells at 0.35, arguing that the compute-optimal frontier requires at least 3B parameters for this benchmark. Their position is grounded in the Chinchilla scaling analysis.',
      'A hardware startup tracking inference costs watches the price as a demand signal — if sub-1B models can hit 90% on MATH, the market for edge inference chips expands significantly. They don\'t trade the conjecture directly, but they use its price to inform their roadmap.',
    ],
    market:
      'Opens around 0.35. As papers demonstrate scaling results, the price moves incrementally. A single definitive result (either achieving or conclusively failing the threshold) pushes the price past 0.95 or below 0.05. Tight bid-ask spread because participants agree on what counts as success or failure — the threshold is unambiguous, so there\'s high consensus on how to interpret any given result, even when there\'s genuine disagreement on the likely outcome. High volume from ML researchers with downstream work.',
  },
  {
    label: 'Bold and testable',
    statement:
      'The majority of gastric ulcers are caused by Helicobacter pylori infection, not by stress, diet, or excess acid production.',
    critiques: [
      'Directly contradicts the prevailing consensus (stress/acid theory), which makes it high-information.',
      'Testable via clinical trials: treat with antibiotics, observe ulcer resolution rates.',
      'The word "majority" introduces interpretive ambiguity — at a population level, "majority" could mean 51% or 90%, and different studies in different populations may yield different rates. This ambiguity should produce some price volatility as participants disagree about whether a given study confirms or undermines the claim.',
    ],
    trades: [
      'Barry Marshall buys at 0.10 based on his own culture experiments. He has direct evidence that the medical establishment hasn\'t seen yet — this is the conjecture market equivalent of an inventor buying equity in their own company. Historically, Marshall was so certain he drank a petri dish of H. pylori to prove his point (1984).',
      'The gastroenterology establishment sells aggressively. They have decades of clinical experience pointing to stress and acid. From their perspective, Marshall is a crank. This is analogous to how the medical establishment dismissed Semmelweis\'s handwashing hypothesis in the 1840s.',
      'A pharmaceutical company exploring broad-spectrum antibiotics for GI conditions quietly accumulates a long position. They don\'t need the conjecture to be at 0.95 to profit — even a move from 0.10 to 0.40 justifies their R&D investment in antibiotic ulcer treatments.',
    ],
    market:
      'Marshall prices it at 0.80. The market sells it down to 0.10 — the establishment disagrees. When Marshall drinks the petri dish and develops gastritis, the price ticks up. Clinical trials showing antibiotic cures push it past 0.90 over the following years. Enormous returns for early believers. Note that the interpretive ambiguity around "majority" creates genuine price volatility along the way — a study showing 60% bacterial causation in one population might push the price to 0.70, while a study in a different population showing only 45% could pull it back to 0.55. The price eventually stabilizes above 0.90 as the weight of evidence across populations becomes overwhelming, but the path there is noisier than a perfectly operationalized conjecture would produce.',
  },
  {
    label: 'Confuses correlation with mechanism',
    statement:
      'Countries with higher per-capita chocolate consumption produce more Nobel laureates per capita.',
    critiques: [
      'This is a real observed correlation (Messerli, 2012). But it\'s stated as a bare association with no mechanism.',
      'It has no predictive power under intervention: if Switzerland doubled its chocolate consumption, it would not produce more Nobel laureates.',
      'The real drivers (wealth, education infrastructure, research funding) are omitted, making the conjecture misleading even though it\'s technically "true."',
    ],
    trades: [
      'A data journalist buys at 0.75 because the correlation genuinely exists in published data. They\'re technically correct, and they\'ll earn a modest return when the price stabilizes.',
      'An epidemiologist ignores it entirely. They recognize the conjecture as unactionable — meaning no intervention based on this claim would produce the predicted outcome. "Actionable" in this context means a participant could use the conjecture to make predictions about what would happen if someone changed the input variable. "If I increase chocolate distribution in Country X, will Nobel production increase?" The answer is obviously no, which reveals the conjecture has no causal content.',
      'A portfolio strategist accumulates dozens of technically-true correlations like this one. Their portfolio looks impressive on paper — many positions, all at high credence (~0.85). But because none of these conjectures generate useful predictions or downstream conjectures, the portfolio has essentially zero delta (sensitivity to new evidence) and zero alpha (returns from insight). It\'s a portfolio of dead weight: occupying capital without generating information. Their veracity consensus flatlines despite the large position count. (This failure mode — large portfolios of trivially true but uninformative conjectures — is an open design problem; see Open Problems.)',
    ],
    market:
      'Price converges to ~0.85 because the correlation genuinely exists in the data. But the conjecture generates no useful downstream conjectures — you can\'t derive policy or predictions from it. Trade volume is low because sophisticated participants recognize it as unactionable. It occupies market space without adding information. Compare to the better version: "National R&D spending as a fraction of GDP is the strongest single predictor of per-capita Nobel laureates" — that version is falsifiable, mechanistic, and yields predictions you could actually test by examining countries that increased R&D spending.',
  },
  {
    label: 'Low predictive capacity (too vague)',
    statement:
      'AI will fundamentally transform society within the next few decades.',
    critiques: [
      'What counts as "fundamentally"? What counts as "transform"? What counts as "society"? And "next few decades" is anywhere from 10 to 40 years.',
      'Nearly every outcome is consistent with this statement. Whether AI automates 5% of jobs or 95%, someone will argue society was "fundamentally transformed."',
      'No specific prediction can be derived from it.',
    ],
    trades: [
      'A tech optimist buys at 0.80, interpreting "fundamentally transform" to mean something like AGI-level disruption. An AI skeptic sells at 0.60, interpreting "fundamentally" as requiring more than incremental automation. They aren\'t disagreeing about the future — they\'re disagreeing about what the words mean. This is the defining symptom of a vague conjecture: participants trade past each other.',
      'A venture capitalist wants to use the price as a signal for investment timing, but can\'t — the price doesn\'t respond meaningfully to any specific development. A new foundation model launch, a major automation deployment, a regulatory crackdown — none of these move the price more than a few points because any outcome is consistent with the statement.',
      'A volatility-focused fund might find this conjecture interesting precisely because of the noise. Wide bid-ask spreads and low price sensitivity create opportunities for market-making profits. But this is a trading strategy, not a knowledge strategy — the fund profits from the conjecture\'s dysfunction, not from its information content.',
    ],
    market:
      'Wide bid-ask spread. Buyers and sellers aren\'t disagreeing about the future — they\'re disagreeing about what the statement means. Low price sensitivity: virtually no paper or product launch would move it, because any evidence can be interpreted either way. Liquidity drains to more specific conjectures like "AI systems will automate >50% of customer service interactions at Fortune 500 companies by 2030." This happens because participants who actually have downstream work depending on AI outcomes — researchers, investors, policymakers — need specific claims to trade against. A vague conjecture doesn\'t let them hedge or express a precise view, so their capital migrates to instruments that do. This conjecture becomes a dead instrument.',
  },
  {
    label: 'Built-in escape hatch',
    statement:
      'Quantum computing will achieve practical advantage over classical computing for optimization problems, unless hardware error rates remain too high or the problems turn out to be classically tractable after all.',
    critiques: [
      'The "unless" clauses absorb every possible counterexample. If quantum computers fail to show advantage, it\'s because error rates were too high. If classical algorithms improve to match, the problems were "classically tractable after all."',
      'There is no outcome that falsifies this conjecture. It is structured to be right regardless of what happens.',
    ],
    trades: [
      'A quantum computing startup buys for signaling purposes — they want to point to a high market price as evidence of community confidence. But sophisticated participants see through this, and the price doesn\'t respond to the purchase because the claim is recognized as unfalsifiable.',
      'A short seller identifies the escape hatch and attempts to profit by shorting. They discover they can never collect — every piece of counter-evidence triggers one of the "unless" clauses, and the price never drops below 0.40 even as classical algorithms improve. Their capital is trapped.',
      'Serious quantum computing researchers ignore this conjecture entirely and trade the clean version: "A quantum computer will solve a commercially deployed optimization problem at least 10x faster than the best known classical algorithm by 2030." That version has the same thematic exposure but with actual resolution criteria.',
    ],
    market:
      'Sophisticated participants avoid it entirely — they recognize there\'s no way to be wrong, which means there\'s no way to be informatively right either. Price drifts aimlessly. Anyone who tries to sell it short discovers they can\'t collect, because the escape hatch always triggers. Liquidity migrates to the clean version because capital needs to earn returns — an untradeable conjecture is dead capital. The clean version offers the same thematic exposure with actual resolution criteria, so participants rationally move their capital to where it can actually be productive. The market\'s own incentive structure selects for well-formed conjectures over time.',
  },
  {
    label: 'Ambiguous operationalization',
    statement:
      'Large language models are capable of genuine reasoning.',
    critiques: [
      'What is "genuine reasoning"? Participants will operationalize this differently: some require formal logic, some accept chain-of-thought performance, some demand novel mathematical proofs, some require understanding (whatever that means).',
      'This isn\'t a disagreement about the world — it\'s a disagreement about definitions. The conjecture conflates an empirical question with a philosophical one.',
    ],
    trades: [
      'A cognitive scientist sells at 0.30, because to them "genuine reasoning" requires causal world models, and they have theoretical arguments for why autoregressive token prediction cannot produce these. Their sell is grounded in Judea Pearl\'s causal inference framework.',
      'An ML engineer buys at 0.70, because to them "genuine reasoning" means solving novel problems via chain-of-thought decomposition, and GPT-4 already does this on competition math. They point to the ARC benchmark results as evidence.',
      'These two participants are not disagreeing about the capabilities of LLMs. They agree on what the models can do. They disagree about what the word "reasoning" means. The market cannot aggregate their views because they are trading different claims under the same name.',
    ],
    market:
      'Extremely wide bid-ask spread — not from uncertainty about LLMs, but from participants having incompatible definitions of "genuine reasoning." When a new capability result is published, half the market says it\'s evidence for and half says it doesn\'t count. Price is noisy and uninformative. Volume is high but chaotic: people are trading past each other. A better version decomposes this into specific, measurable sub-claims: "GPT-5 will solve >60% of novel competition math problems" or "An LLM will generate a proof accepted by the Lean proof assistant for a previously unproven theorem."',
  },
  {
    label: 'Generative, spawns sub-conjectures',
    statement:
      'Massive objects curve spacetime, and the trajectory of objects through curved spacetime produces the effects we observe as gravity. Newtonian gravity is the low-mass, low-velocity approximation.',
    critiques: [
      'Makes specific, novel predictions: light bending, gravitational time dilation, frame dragging.',
      'Each prediction becomes its own sub-conjecture with independent testability.',
      'Specifies exactly when the predecessor theory breaks down.',
    ],
    trades: [
      'Arthur Eddington buys the light-bending sub-conjecture in 1917, two years before his eclipse expedition to Pr\u00edncipe. He stakes his professional reputation and secures Royal Astronomical Society funding. His early position reflects genuine scientific conviction based on the mathematical elegance of Einstein\'s field equations.',
      'Astronomers working on Mercury\'s perihelion precession trade that sub-conjecture independently. The anomalous 43 arcseconds per century had been a known problem since Le Verrier in 1859. General relativity\'s exact prediction of this value moves the sub-conjecture price sharply, which in turn pushes the parent conjecture upward through the attribution graph.',
      'Decades later, experimental physicists trade the frame-dragging sub-conjecture. Gravity Probe B (launched 2004, results 2011) provides direct measurement. The long delay between the parent conjecture (1915) and this particular confirmation (2011) demonstrates how generative conjectures create trading activity across generations of researchers.',
    ],
    market:
      'High volume because nearly all of physics is downstream. Each sub-conjecture (light bending, perihelion precession, frame dragging) trades actively. Eddington\'s 1919 eclipse results move the price sharply. Tight spread — the math is precise enough that participants agree on what counts as confirmation. The complexity here is in the branching structure: one parent conjecture generates an expanding tree of tradeable sub-conjectures, each with its own price history and evidence base.',
  },
  {
    label: 'Unbounded time horizon',
    statement:
      'A general-purpose quantum computer with >1 million logical qubits will be built.',
    critiques: [
      'No deadline. You might think this can never be falsified — you can always say "not yet."',
      'But the market doesn\'t need a binary resolution to be useful.',
    ],
    trades: [
      'A long-horizon deep-tech fund holds a position at 0.40, treating it as a decades-scale bet. They accept that their capital may be locked for years, but the potential returns from being right early on a transformative technology justify the opportunity cost. This is analogous to how patient capital in venture investing works.',
      'A hardware researcher at a superconducting qubit lab trades the sub-conjectures with deadlines — "IBM will demonstrate >1,000 logical qubits by 2030" — rather than the open-ended parent. They have operational knowledge that makes short-horizon sub-conjectures more tradeable for them.',
      'A decoherence skeptic sells at 0.35, arguing that fundamental physical limits on error correction make scaling past 10,000 logical qubits impossible. Their position is grounded in specific thermodynamic arguments about heat dissipation at scale.',
    ],
    market:
      'Price opens around 0.40. Each hardware advance (better error correction, more physical qubits, new qubit architectures) nudges the price incrementally. Each failure or fundamental obstacle nudges it down. The price is a live-updated community credence, and it\'s informative even if the conjecture never formally resolves. Participants with long time horizons can hold positions; those who need shorter horizons trade the sub-conjectures ("IBM will demonstrate >1,000 logical qubits by 2030"). The market handles this naturally through time-discounting. An interesting edge case: what happens if such a computer is built, used for a few years, and then the technology is abandoned — no one ever builds another? The conjecture is technically true (it was built), but participants who bought expecting sustained quantum computing infrastructure would have been right about the letter of the claim and wrong about its spirit. This highlights why precise language matters: "will be built" is different from "will be built and remain in continuous operation."',
  },
  {
    label: 'Deep mathematical uncertainty',
    statement:
      'P \u2260 NP: there exist problems whose solutions can be verified in polynomial time but cannot be solved in polynomial time.',
    critiques: [
      'Formally precise. One of the seven Clay Millennium Prize Problems, open since Cook (1971) and Karp (1972). Enormous implications for cryptography, optimization, algorithm design, and the foundations of mathematics.',
      'Unlike vague conjectures, the ambiguity here is not in the statement — it is perfectly well-defined. The uncertainty is genuine: humanity does not know the answer.',
      'Partial progress is possible and observable. Results in circuit complexity, barrier theorems, and natural proofs provide incremental evidence without resolving the question.',
    ],
    trades: [
      'A cryptographer holds a long position at 0.95 because their entire security infrastructure assumes P\u2260NP. If the conjecture fell to 0.50, they would need to begin migrating to information-theoretic security schemes. The price functions as a risk indicator for applied cryptography. This is analogous to how insurers monitor climate model credences.',
      'A complexity theorist who believes relativization and natural-proof barriers are close to being circumvented buys aggressively at 0.93, expecting new barrier results to push confidence higher. Their trade is grounded in Ryan Williams\'s circuit lower bound results (2010s) which showed that NEXP \u2289 ACC\u2070, the first progress in decades.',
      'When Vinay Deolalikar circulated a claimed proof of P\u2260NP in August 2010, the price would have spiked briefly before reverting as the proof\'s flaws were identified within weeks by the theory community. The spike-and-revert pattern is characteristic of open problems: claimed proofs surface periodically and are almost always wrong, but each one generates a burst of trading activity and a temporary price disruption.',
    ],
    market:
      'Price sits around 0.95, reflecting overwhelming expert consensus that P\u2260NP, but with a persistent 5% discount reflecting the genuine possibility that we\'re wrong. The price is not static — it responds to structural results in complexity theory. Barrier theorems (Baker-Gill-Solovay relativization, Razborov-Rudich natural proofs) moved the price by clarifying what proof techniques cannot work, paradoxically increasing confidence by narrowing the space. Sub-conjectures about specific complexity class separations (e.g., "NEXP \u2289 P/poly") trade more actively and at lower credences. The long time horizon is not a problem — the price is informative today even if resolution is decades away.',
  },
  {
    label: 'Formally independent of standard axioms',
    statement:
      'The Continuum Hypothesis is true: there is no set whose cardinality is strictly between that of the integers and the real numbers.',
    critiques: [
      'This is not a vague or poorly formed conjecture. It is mathematically precise and was Hilbert\'s first problem (1900).',
      'G\u00f6del (1940) proved it is consistent with ZFC. Cohen (1963) proved its negation is also consistent with ZFC. It is formally independent of the standard axioms of set theory.',
      'No amount of mathematical evidence within ZFC can move the price in either direction, because both the statement and its negation are compatible with the axioms. This is fundamentally different from "we don\'t know yet" — it\'s "the standard framework cannot decide this."',
    ],
    trades: [
      'A set theorist working in forcing theory recognizes the independence result and avoids trading entirely. They know that no proof within ZFC can arrive, so any price movement is noise, not signal.',
      'A philosopher of mathematics takes a position at 0.60 based on "mathematical naturalism" arguments — they believe large cardinal axioms that settle the question in one direction reflect genuine mathematical reality. Their trade is grounded in G\u00f6del\'s own view that the Continuum Hypothesis is likely false, and in Woodin\'s more recent arguments for new axioms.',
      'A logician attempts to short at 0.55, arguing that the independence result means the price should converge to 0.50 (maximum uncertainty). But there\'s no counterparty mechanism to enforce convergence, because there\'s no evidence that could prove them right.',
    ],
    market:
      'The price wanders between 0.40 and 0.60 based on which foundational framework participants prefer, not based on evidence. Volume is very low because sophisticated participants recognize it as untradeable within standard mathematics. The spread is wide and persistent. This is the most interesting failure mode in the catalog: the conjecture is perfectly well-formed, mathematically precise, and was posed by one of the greatest mathematicians in history — yet the market breaks anyway. Not because of vagueness or escape hatches, but because the underlying question is formally independent of the axiom system that participants share. The price reflects philosophical taste, not empirical or mathematical credence. It raises a deep design question: should the market flag known-independent statements, or let participants discover the independence through trading dynamics?',
  },
]

export default function ExampleConjecturesPage() {
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
            Example Conjectures
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            A catalog of conjectures&mdash;good, bad, and misleading&mdash;with
            market analysis for each. Ordered from simplest price dynamics to most
            complex. The goal is not to memorize rules for what makes a good
            conjecture, but to develop intuition for how the market responds to
            different kinds of claims.
          </p>
        </header>

        <div className="space-y-12" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          <div className="space-y-4">
            {examples.map((ex, i) => (
              <ExampleCard key={i} example={ex} />
            ))}
          </div>

          {/* Summary */}
          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              The test
            </h2>
            <p className="leading-relaxed mb-4">
              The question to ask about any conjecture is not &ldquo;is it true?&rdquo; or
              even &ldquo;is it testable?&rdquo; but: <strong>can the market process it?</strong>
            </p>
            <p className="leading-relaxed mb-4">
              A conjecture the market can process has a tight spread (participants agree on
              what it means), price sensitivity to evidence (new results move the price),
              and volume (people have downstream work that depends on it). A conjecture the
              market cannot process has a wide spread, noisy price, and chaotic volume where
              participants are effectively trading different claims under the same name.
            </p>
            <p className="leading-relaxed">
              Wrong conjectures, boring conjectures, and open-ended conjectures are all fine.
              Vague conjectures, tautological conjectures, escape-hatch conjectures, and
              ambiguously operationalized conjectures are the ones that break the market. And
              at the far edge, formally undecidable conjectures reveal the deepest limitation:
              even a perfectly well-formed statement can defeat the market if the shared axiom
              system cannot resolve it. The common thread: they prevent the price signal from
              doing its job.
            </p>
          </section>

          <section>
            <h2
              className="text-xl font-bold mb-4"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Informal quality scorecard
            </h2>
            <p className="leading-relaxed mb-4">
              The following scorecard is not enforced or calculated by the
              system. It is an informal heuristic &mdash; a way for
              participants to quickly assess whether a conjecture is likely
              to be useful as a market instrument before they invest time or
              capital. Think of it as a checklist, not a formula.
            </p>
            <ul className="space-y-3 ml-6 list-disc">
              <li className="leading-relaxed">
                <strong>Entropy sensitivity to evidence.</strong> Does new
                evidence move the credence? Tautological and unfalsifiable
                conjectures score zero.
              </li>
              <li className="leading-relaxed">
                <strong>Spread width.</strong> Tight means participants
                agree on what the conjecture means. Wide often signals
                definitional ambiguity, not genuine uncertainty.
              </li>
              <li className="leading-relaxed">
                <strong>Volume and diversity.</strong> High volume from
                diverse participants means downstream relevance. Low or
                concentrated volume means isolation.
              </li>
              <li className="leading-relaxed">
                <strong>Downstream conjecture count.</strong> Does it spawn
                sub-conjectures? Zero downstream connections means a dead
                end.
              </li>
              <li className="leading-relaxed">
                <strong>Resolution criteria clarity.</strong> Can
                participants agree on what evidence would move the credence
                to 0.95 or 0.05?
              </li>
              <li className="leading-relaxed">
                <strong>Entropy contribution to portfolio.</strong> Does
                holding this add remaining uncertainty to your portfolio, or
                is it dead weight?
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
                Worked examples
              </h3>
              <p className="leading-relaxed mb-4">
                <strong>&ldquo;In efficient markets, all available
                information is reflected in asset prices.&rdquo;</strong>{' '}
                Entropy sensitivity 0/10, spread 9/10 (wide), volume 1/10,
                downstream 0, resolution clarity 0/10, entropy contribution
                0. A tautology with no market value.
              </p>
              <p className="leading-relaxed">
                <strong>&ldquo;Transformer models with fewer than 1B
                parameters can achieve &gt;90% accuracy on
                MATH.&rdquo;</strong>{' '}
                Entropy sensitivity 8/10, spread 2/10 (tight), volume 7/10,
                downstream 4+, resolution clarity 9/10, entropy contribution
                7/10. A well-formed conjecture that the market can process.
              </p>
            </div>
          </section>

        </div>
      </div>
    </div>
  )
}
