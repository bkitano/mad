import { Link } from 'react-router-dom'

interface Example {
  label: string
  statement: string
  critiques: string[]
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

const goodExamples: Example[] = [
  {
    label: 'Good — specific and falsifiable',
    statement:
      'Transformer models with fewer than 1B parameters can achieve >90% accuracy on MATH benchmark problems when trained with verifier-guided search.',
    critiques: [
      'Clear threshold (1B params, 90% accuracy), specific benchmark (MATH), specific method (verifier-guided search).',
      'Any team can attempt to reproduce it. Success or failure moves the price directly.',
    ],
    market:
      'Opens around 0.35. As papers demonstrate scaling results, the price moves incrementally. A single definitive result (either achieving or conclusively failing the threshold) resolves it. Tight bid-ask spread because participants agree on what counts. High volume from ML researchers with downstream work.',
  },
  {
    label: 'Good — bold and testable',
    statement:
      'The majority of gastric ulcers are caused by Helicobacter pylori infection, not by stress, diet, or excess acid production.',
    critiques: [
      'Directly contradicts the prevailing consensus (stress/acid theory), which makes it high-information.',
      'Testable via clinical trials: treat with antibiotics, observe ulcer resolution rates.',
    ],
    market:
      'Marshall prices it at 0.80. The market sells it down to 0.10 — the establishment disagrees. When Marshall drinks the petri dish and develops gastritis, the price ticks up. Clinical trials showing antibiotic cures push it past 0.90 over the following years. Enormous returns for early believers.',
  },
  {
    label: 'Good — generative, spawns sub-conjectures',
    statement:
      'Massive objects curve spacetime, and the trajectory of objects through curved spacetime produces the effects we observe as gravity. Newtonian gravity is the low-mass, low-velocity approximation.',
    critiques: [
      'Makes specific, novel predictions: light bending, gravitational time dilation, frame dragging.',
      'Each prediction becomes its own sub-conjecture with independent testability.',
      'Specifies exactly when the predecessor theory breaks down.',
    ],
    market:
      'High volume because nearly all of physics is downstream. Each sub-conjecture (light bending, perihelion precession) trades actively. Eddington\'s 1919 eclipse results move the price sharply. Tight spread — the math is precise enough that participants agree on what counts as confirmation.',
  },
]

const badExamples: Example[] = [
  {
    label: 'Bad — low predictive capacity (too vague)',
    statement:
      'AI will fundamentally transform society within the next few decades.',
    critiques: [
      'What counts as "fundamentally"? What counts as "transform"? What counts as "society"? And "next few decades" is anywhere from 10 to 40 years.',
      'Nearly every outcome is consistent with this statement. Whether AI automates 5% of jobs or 95%, someone will argue society was "fundamentally transformed."',
      'No specific prediction can be derived from it.',
    ],
    market:
      'Wide bid-ask spread. Buyers and sellers aren\'t disagreeing about the future — they\'re disagreeing about what the statement means. Low price sensitivity: virtually no paper or product launch would move it, because any evidence can be interpreted either way. Liquidity drains to more specific conjectures like "AI systems will automate >50% of customer service interactions at Fortune 500 companies by 2030." This conjecture becomes a dead instrument.',
  },
  {
    label: 'Bad — low predictive capacity (doesn\'t generalize)',
    statement:
      'GPT-4 scores 86.4% on the MMLU benchmark.',
    critiques: [
      'This is a measurement, not a conjecture. It describes a single, already-observed data point.',
      'It generates no predictions about anything else. It tells you nothing about GPT-5, nothing about other benchmarks, nothing about capability in general.',
      'There\'s nothing to trade against because there\'s no uncertainty about the future.',
    ],
    market:
      'Price immediately converges to ~1.0 (or ~0.0 if wrong). No trade volume after initial convergence because there\'s nothing left to learn. No downstream conjectures — it\'s a dead end. Identical to betting on a coin flip that already happened.',
  },
  {
    label: 'Bad — tautological / unfalsifiable',
    statement:
      'In efficient markets, all available information is reflected in asset prices.',
    critiques: [
      'If you define "efficient" as "all information is reflected in prices," this is a tautology — true by definition.',
      'Any apparent mispricing is explained away: either the market wasn\'t efficient in that instance, or you don\'t have the same information the market does.',
      'No possible observation can count as evidence against it, because the escape hatch is built into the definition.',
    ],
    market:
      'The market cannot process this conjecture. No evidence moves the price, so the price is arbitrary and stays wherever it was initially set. Zero trade volume — sophisticated participants recognize it as untradeable. Wide spread because the few participants who do trade are essentially guessing about a meaningless number.',
  },
  {
    label: 'Bad — trivially narrow',
    statement:
      'The next paper published by the AlphaFold team will have more than 12 authors.',
    critiques: [
      'Technically falsifiable — you can check the author list. But it generates zero scientific insight.',
      'Knowing the answer tells you nothing about protein folding, team dynamics in general, or anything else.',
      'It has no downstream conjectures. Nothing depends on it.',
    ],
    market:
      'Very low volume. No researcher has work that depends on this claim. Price sensitivity is binary — it resolves exactly once, and no intermediate evidence updates it meaningfully. The market treats it as noise. Even if you get it right, your portfolio learns nothing from it — it doesn\'t improve your veracity score on anything that matters.',
  },
  {
    label: 'Bad — built-in escape hatch',
    statement:
      'Quantum computing will achieve practical advantage over classical computing for optimization problems, unless hardware error rates remain too high or the problems turn out to be classically tractable after all.',
    critiques: [
      'The "unless" clauses absorb every possible counterexample. If quantum computers fail to show advantage, it\'s because error rates were too high. If classical algorithms improve to match, the problems were "classically tractable after all."',
      'There is no outcome that falsifies this conjecture. It is structured to be right regardless of what happens.',
    ],
    market:
      'Sophisticated participants avoid it entirely — they recognize there\'s no way to be wrong, which means there\'s no way to be informatively right either. Price drifts aimlessly. Anyone who tries to sell it short discovers they can\'t collect, because the escape hatch always triggers. Liquidity migrates to the clean version: "A quantum computer will solve a commercially deployed optimization problem at least 10x faster than the best known classical algorithm by 2030."',
  },
  {
    label: 'Bad — ambiguous operationalization',
    statement:
      'Large language models are capable of genuine reasoning.',
    critiques: [
      'What is "genuine reasoning"? Participants will operationalize this differently: some require formal logic, some accept chain-of-thought performance, some demand novel mathematical proofs, some require understanding (whatever that means).',
      'This isn\'t a disagreement about the world — it\'s a disagreement about definitions. The conjecture conflates an empirical question with a philosophical one.',
    ],
    market:
      'Extremely wide bid-ask spread — not from uncertainty about LLMs, but from participants having incompatible definitions of "genuine reasoning." When a new capability result is published, half the market says it\'s evidence for and half says it doesn\'t count. Price is noisy and uninformative. Volume is high but chaotic: people are trading past each other. A better version decomposes this into specific, measurable sub-claims: "GPT-5 will solve >60% of novel competition math problems" or "An LLM will generate a proof accepted by the Lean proof assistant for a previously unproven theorem."',
  },
  {
    label: 'Bad — confuses correlation with mechanism',
    statement:
      'Countries with higher per-capita chocolate consumption produce more Nobel laureates per capita.',
    critiques: [
      'This is a real observed correlation (Messerli, 2012). But it\'s stated as a bare association with no mechanism.',
      'It has no predictive power under intervention: if Switzerland doubled its chocolate consumption, it would not produce more Nobel laureates.',
      'The real drivers (wealth, education infrastructure, research funding) are omitted, making the conjecture misleading even though it\'s technically "true."',
    ],
    market:
      'Price converges to ~0.85 because the correlation genuinely exists in the data. But the conjecture generates no useful downstream conjectures — you can\'t derive policy or predictions from it. Trade volume is low because sophisticated participants recognize it as unactionable. It occupies market space without adding information. Compare to the better version: "National R&D spending as a fraction of GDP is the strongest single predictor of per-capita Nobel laureates" — that version is falsifiable, mechanistic, and actionable.',
  },
]

const surprisinglyFineExamples: Example[] = [
  {
    label: 'Seems bad, actually fine — provably false',
    statement:
      'Heavier objects fall faster than lighter objects in a vacuum.',
    critiques: [
      'This is straightforwardly wrong. Galileo demonstrated it in the 1600s. Every physics student can verify it.',
      'You might think this wastes the market\'s time — but it doesn\'t.',
    ],
    market:
      'Price immediately crashes to ~0.01. This is the market working correctly. A provably false conjecture is a free lunch for informed participants: they short it, collect easy returns, and the price signal broadcasts "this is wrong" to anyone watching. The market processes it quickly and efficiently. The conjecture is bad science, but it\'s a perfectly fine market instrument — it just resolves fast.',
  },
  {
    label: 'Seems bad, actually fine — already consensus',
    statement:
      'Water boils at 100°C at standard atmospheric pressure (1 atm).',
    critiques: [
      'This is already universally known. You might think it has no market value — price is already at 1.0, no trading opportunity.',
    ],
    market:
      'Price sits at ~0.99 with a razor-thin spread. Virtually no volume. But this is fine — the market is correctly reflecting that this is settled knowledge. Not every conjecture needs to be controversial to be valid. It serves as an anchor: if someone proposes a conjecture that implies water doesn\'t boil at 100°C, the market can immediately price that new conjecture down by referencing this one. Settled conjectures are the bedrock of the market\'s knowledge graph.',
  },
  {
    label: 'Seems bad, actually fine — unbounded time horizon',
    statement:
      'A general-purpose quantum computer with >1 million logical qubits will be built.',
    critiques: [
      'No deadline. You might think this can never be falsified — you can always say "not yet."',
      'But the market doesn\'t need a binary resolution to be useful.',
    ],
    market:
      'Price opens around 0.40. Each hardware advance (better error correction, more physical qubits, new qubit architectures) nudges the price incrementally. Each failure or fundamental obstacle nudges it down. The price is a live-updated community credence, and it\'s informative even if the conjecture never formally resolves. Participants with long time horizons can hold positions; those who need shorter horizons trade the sub-conjectures ("IBM will demonstrate >1000 logical qubits by 2030"). The market handles this naturally through time-discounting.',
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
            market analysis for each. The goal is not to memorize rules for what
            makes a good conjecture, but to develop intuition for how the market
            responds to different kinds of claims.
          </p>
        </header>

        <div className="space-y-12" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* Good examples */}
          <section>
            <h2
              className="text-xl font-bold mb-6"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Good conjectures
            </h2>
            <p className="leading-relaxed mb-6">
              Good conjectures are specific, falsifiable, and generative. They say one
              thing clearly enough that evidence can move the price, and they connect to
              other questions in ways that make them worth trading.
            </p>
            <div className="space-y-4">
              {goodExamples.map((ex, i) => (
                <ExampleCard key={i} example={ex} />
              ))}
            </div>
          </section>

          {/* Bad examples */}
          <section>
            <h2
              className="text-xl font-bold mb-6"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Bad conjectures
            </h2>
            <p className="leading-relaxed mb-6">
              Bad conjectures are ones the market cannot process well. The price signal
              becomes noisy, the spread widens, volume dies, or participants talk past
              each other. The failure isn&rsquo;t that the conjecture is wrong&mdash;wrong
              conjectures are fine. The failure is that the market can&rsquo;t do its job.
            </p>
            <div className="space-y-4">
              {badExamples.map((ex, i) => (
                <ExampleCard key={i} example={ex} />
              ))}
            </div>
          </section>

          {/* Surprisingly fine */}
          <section>
            <h2
              className="text-xl font-bold mb-6"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Surprisingly fine conjectures
            </h2>
            <p className="leading-relaxed mb-6">
              Some conjectures seem bad at first glance but are actually handled well by
              the market. The key distinction: a conjecture is only truly bad if the market
              can&rsquo;t process it. If the market can price it correctly&mdash;even if the
              price is boring or the resolution is instant&mdash;the conjecture is fine.
            </p>
            <div className="space-y-4">
              {surprisinglyFineExamples.map((ex, i) => (
                <ExampleCard key={i} example={ex} />
              ))}
            </div>
          </section>

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
              ambiguously operationalized conjectures are the ones that break the market. The
              common thread: they prevent the price signal from doing its job.
            </p>
          </section>

        </div>
      </div>
    </div>
  )
}
