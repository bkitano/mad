# Minimum Viable Science Market: ML Domain

An RL environment where agents trade on scientific conjectures reconstructed from the machine learning literature. The goal is to train agents that develop genuine scientific judgment — calibration, evidence discrimination, and efficient inquiry — by betting on claims extracted from ML papers, scored against what actually happened.

## 1. Reconstructing the Science Market

### Source data

The ML domain from 2020 (GPT-3) to 2024. Primary sources:

- **Semantic Scholar API** — paper metadata, abstracts, citation graphs, publication dates
- **arXiv bulk access** — full text for ML papers (cs.LG, cs.CL, cs.CV, stat.ML)
- **Papers With Code** — links papers to benchmarks, datasets, and reported results (SOTA tables)
- **Retraction Watch** — retracted or corrected ML papers

### Conjecture extraction

A conjecture is a falsifiable claim extracted from a paper. We extract them at three levels of granularity:

**Level 1: SOTA claims.** "ResNet achieves < 4% top-5 error on ImageNet." These are the easiest — Papers With Code already tracks them. Each SOTA claim is a conjecture that starts at credence ~0.5 when the paper is published (the community hasn't verified it yet) and resolves as replications succeed or fail. There are thousands of these.

**Level 2: Methodological claims.** "Batch normalization reduces internal covariate shift." "Dropout is equivalent to approximate Bayesian inference." "Scaling laws follow power-law relationships." These are extracted from paper abstracts and introductions using an LLM with the prompt: *"Extract the core falsifiable claims from this abstract. Each claim should be a statement that could be true or false, not a description of what the paper does."* These take longer to resolve and have richer evidence trails.

**Level 3: Paradigm claims.** "Scaling is more important than architecture." "RLHF aligns language models with human preferences." "In-context learning is implicit fine-tuning." These are the high-entropy, long-lived conjectures that generate the most interesting market dynamics. They're extracted from survey papers, blog posts, and position papers. There are fewer of them (maybe 30-50 major ones across the period) but they're the backbone of the dependency graph.

### Building the time series

For each conjecture, we construct a timeline of evidence events:

```
conjecture: "RLHF is sufficient to align language models with human values"
created: 2022-01 (Ouyang et al., InstructGPT)
events:
  - 2022-04: paper supporting (Bai et al., Anthropic shows RLHF reduces harmful outputs)
  - 2022-12: paper supporting (ChatGPT launch — massive real-world validation of RLHF usability)
  - 2023-05: paper contradicting (Casper et al., open problems in RLHF — reward hacking, distributional shift)
  - 2023-07: paper contradicting (Wolf et al., jailbreaks bypass RLHF safety training trivially)
  - 2023-10: paper supporting (Touvron et al., Llama 2 shows RLHF scales to open models)
  - 2024-02: paper contradicting (Anthropic, Sleeper Agents — RLHF doesn't remove deceptive behaviors)
  - 2024-05: paper extending (Rafailov et al., DPO simplifies RLHF but shares its limitations)
resolution: contested (credence ~0.35 — RLHF helps but "sufficient" is increasingly doubted)
```

Each evidence event includes:
- **Timestamp** (publication date)
- **Direction** (supports, contradicts, or extends the conjecture)
- **Strength signal** (citation velocity in the 12 months after publication — a proxy for how seriously the community took it)
- **Full text** (the abstract + introduction, used as the observation the agent reads)

### Dependency graph

Conjectures are linked by logical dependency, extracted from citation patterns:

- If paper B cites paper A's core claim as a premise, conjecture B depends on conjecture A
- Co-citation clusters reveal implicit dependencies (conjectures frequently cited together likely share assumptions)
- LLM-assisted: given two conjectures, "Does the truth of A make B more or less likely? Or are they independent?" — used to fill gaps in citation-derived structure

The dependency graph for ML 2020-2024 has natural clusters: scaling laws, alignment/RLHF, in-context learning, multimodal models, open-source vs. closed, efficiency/distillation, etc.

### Pricing

Each conjecture is priced using the entropy-based mechanism from the platform:

- **Credence** `P(A)`: the market's current belief, updated by evidence events
- **Cost (YES)** `P(A) * H(P)`: credence times entropy — expensive to agree with the crowd at high uncertainty, cheap to be contrarian
- **Cost (NO)** `(1 - P(A)) * H(P)`: the mirror — cheap to disagree when credence is high
- **Position value** `r = H(P_entry) * (P_now - P_entry)` for YES; reversed for NO

The direction-weighted cost creates the right incentive structure: it's always cheaper to challenge the orthodoxy than to pile on. At P=0.50, both sides cost the same (0.50 * 1.0 = 0.50). As credence shifts, the contrarian side gets cheaper. Think of it as a 2x2:

|  | High uncertainty (H ≈ 1.0) | Low uncertainty (H ≈ 0.3) |
|---|---|---|
| **High credence (P = 0.8)** | YES=0.80, NO=0.20. Contrarian NO is cheap. | YES=0.24, NO=0.06. Everything cheap, resolved. |
| **Low credence (P = 0.2)** | YES=0.20, NO=0.80. Contrarian YES is cheap. | YES=0.06, NO=0.24. Pennies to take a flier. |

Credence updates use a simplified Bayesian update: each evidence event shifts credence by an amount proportional to the evidence strength signal, decayed by how many prior evidence events exist (diminishing marginal evidence).

## 2. The RL Environment

### State space

At each timestep `t`, the agent observes:

```python
state = {
    # The conjecture being evaluated
    "conjecture_text": str,          # the claim in natural language
    "current_credence": float,       # P(A) at time t
    "current_entropy": float,        # H(P) at time t
    "time_since_creation": int,      # months since conjecture was proposed

    # Evidence history (all evidence seen so far, chronologically)
    "evidence_history": [
        {
            "text": str,             # abstract + intro of the paper
            "direction": str,        # "supports" | "contradicts" | "extends"
            "strength": float,       # citation velocity proxy, normalized 0-1
            "age": int,              # months since this evidence was published
        },
        ...
    ],

    # New evidence arriving this timestep
    "new_evidence": {
        "text": str,
        "direction": str,
        "strength": float,
    } | None,

    # Portfolio state
    "budget_remaining": float,       # remaining capital
    "current_positions": [           # agent's open positions
        {
            "conjecture_id": str,
            "direction": str,        # "YES" | "NO"
            "shares": int,
            "entry_credence": float,
            "entry_entropy": float,
        },
        ...
    ],

    # Market context (other active conjectures the agent can trade)
    "available_conjectures": [
        {
            "conjecture_id": str,
            "conjecture_text": str,
            "current_credence": float,
            "current_entropy": float,
            "evidence_count": int,
            "dependency_ids": [str],
        },
        ...
    ],
}
```

### Action space

At each timestep, the agent takes one action:

```python
action = {
    "type": str,  # "buy_yes" | "buy_no" | "sell" | "hold"

    # Required for buy_yes / buy_no:
    "conjecture_id": str,
    "shares": int,             # number of shares (cost = shares * P_dir * H(P))

    # Required for sell:
    "position_index": int,     # which position to close
}
```

The action space is discrete: choose a conjecture, choose a direction, choose a size. The `hold` action is always available and costs nothing.

### Timestep mechanics

The environment steps monthly through the reconstructed timeline:

1. **Reveal new evidence.** Any papers published this month are shown to the agent as `new_evidence` for the relevant conjectures.
2. **Agent acts.** The agent observes the updated state and takes an action.
3. **Market updates.** Credences update based on the new evidence (not the agent's trade — the agent is a price-taker, not a market-maker). Entropy recomputes.
4. **Reward computes.** The agent receives the change in portfolio value this timestep: `sum over positions of (shares * H_entry * delta_credence)`.

### Episode structure

An episode is one domain slice:

- **Training episodes:** Random 18-month windows from 2020-2023, with 20-50 active conjectures per window. Agent starts with a fixed budget and empty portfolio.
- **Validation episodes:** Fixed 18-month windows from mid-2022 to end-2023, same conjectures but held-out time periods. Checks temporal generalization.
- **Test episodes:** 2024. Never seen during training. Final benchmark scores computed here.

### Reward function

```
R_t = delta_portfolio_value_t + lambda * calibration_bonus_t
```

Where:
- `delta_portfolio_value_t` = change in total portfolio value this timestep
- `calibration_bonus_t` = small bonus for well-calibrated confidence (optional, can be ablated)
- `lambda` = weighting hyperparameter (default 0.1)

The portfolio value already encodes the entropy weighting, so early high-conviction bets on uncertain conjectures naturally produce the highest returns.

### Reset and parallelization

Episodes are independent 36-month windows. Thousands can run in parallel. Each episode takes seconds to step through (reading text is the bottleneck, not computation). A single GPU can run ~100 episodes/hour if the agent is an LLM; a lightweight policy network can run ~10,000 episodes/hour.

## 3. The Benchmark

### Headline metric

**Science Market Return (SMR):** Portfolio return on the held-out test set (2022-2024), computed as total portfolio value at episode end divided by starting budget. Reported as a percentage. Higher is better.

### Diagnostic metrics

| Metric | What it measures | How it's computed | Unit |
|---|---|---|---|
| **Calibration (Brier)** | Probability accuracy | Mean squared error of credence predictions vs. outcomes | 0-1, lower is better |
| **Early Discovery (Alpha)** | Timing of correct bets | Mean evidence-events-ahead-of-consensus for correct positions | Events, higher is better |
| **Evidence Discrimination** | Evidence quality judgment | Accuracy on forced-choice pairs: robust vs. misleading evidence | %, higher is better |
| **Dependency F1** | Structural reasoning | F1 of recovered dependency edges vs. ground-truth graph | 0-1, higher is better |
| **Budget Efficiency** | Information per dollar | Final Brier score improvement per unit of budget consumed | Score/dollar, higher is better |

### Baselines

Every benchmark needs baselines to be interpretable:

- **Random agent:** Buys random positions with random sizing. Establishes the floor.
- **Majority agent:** Always bets with the current credence direction. Buys YES when credence > 0.5, NO when < 0.5. Tests whether the market is already efficient.
- **Citation-count agent:** Bets proportional to citation velocity of new evidence. Tests whether a simple heuristic captures most of the signal.
- **Zero-shot LLM agent:** An LLM (GPT-4, Claude, etc.) reads the evidence and predicts credence with no RL training. Tests whether the market structure adds value over prompting.
- **Human expert baseline:** ML researchers (us) manually trade on 50 held-out conjectures. Establishes the ceiling and sanity-checks the benchmark.

### Leaderboard format

```
Rank | Agent           | SMR (%) | Brier | Alpha | Ev.Disc | Dep.F1 | Eff.
-----|-----------------|---------|-------|-------|---------|--------|------
  1  | [agent name]    |  +34.2  | 0.18  |  4.2  |  78.3%  |  0.61  | 0.42
  2  | Human expert    |  +28.7  | 0.21  |  3.8  |  82.1%  |  0.54  | 0.38
  3  | Zero-shot LLM   |  +15.4  | 0.27  |  1.9  |  71.2%  |  0.32  | 0.21
  4  | Citation agent  |  +11.0  | 0.31  |  2.1  |  58.0%  |  0.15  | 0.19
  5  | Majority agent  |   +3.2  | 0.34  |  0.0  |  50.0%  |  0.00  | 0.05
  6  | Random agent    |   -8.1  | 0.50  | -0.3  |  50.0%  |  0.00  | 0.01
```

## 4. Running Agents Against the Benchmark

### Integration interface

Agents implement a single function:

```python
class ScienceMarketAgent:
    def act(self, state: MarketState) -> MarketAction:
        """Given the current market state, return an action."""
        ...
```

The environment handles everything else: stepping time, updating credences, computing portfolio value, tracking metrics.

### Running an evaluation

```bash
# Run a single agent on the test set
python -m market.evaluate --agent my_agent.MyAgent --split test

# Run with specific config
python -m market.evaluate \
    --agent my_agent.MyAgent \
    --split test \
    --budget 1000 \
    --episode-length 36 \
    --num-episodes 100

# Compare multiple agents
python -m market.compare \
    --agents random,majority,citation,zero_shot_llm,my_agent.MyAgent \
    --split test \
    --output results/comparison.json
```

### RL training loop

For agents that learn via RL:

```python
from market.env import ScienceMarketEnv

env = ScienceMarketEnv(
    domain="ml",
    split="train",
    episode_length=36,      # months
    budget=1000,
    max_conjectures=50,     # active conjectures per episode
)

state = env.reset()
done = False

while not done:
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    agent.learn(state, action, reward)  # your RL algorithm here
```

The environment follows the Gymnasium API. Compatible with standard RL libraries (Stable Baselines3, CleanRL, RLlib).

### LLM agent wrapper

For LLM-based agents that reason in natural language:

```python
class LLMScienceAgent(ScienceMarketAgent):
    def act(self, state: MarketState) -> MarketAction:
        prompt = self.format_state(state)  # convert state to text
        response = self.llm.generate(prompt)
        return self.parse_action(response)  # extract structured action
```

The benchmark ships with a reference LLM agent that formats state as a structured prompt and parses JSON actions from the response.

## 5. Faithfulness of the Backtest

The central question: does performance on this simulated market predict anything about an agent's ability to reason about real, open scientific questions? Here is how we validate faithfulness.

### Sanity checks (must pass)

These are necessary conditions. If any fail, the simulation is broken:

- **Credence convergence.** Conjectures that the ML community accepted (e.g., "chain-of-thought prompting improves reasoning performance") must reach credence > 0.85 by the end of the timeline. Conjectures that were rejected or abandoned must reach < 0.15. If the market doesn't converge on known outcomes, the evidence pipeline is failing.
- **Entropy monotonicity.** For conjectures with accumulating evidence, entropy should generally decrease over time. Sustained entropy increases without new contradictory evidence indicate a broken update mechanism.
- **Reward face validity.** Hindsight portfolio rankings should make intuitive sense. An agent that bet YES on "scaling laws" early should outperform one that bet NO. If the reward function produces counterintuitive rankings on well-known historical cases, the mechanism is wrong.

### Calibration consistency

- **Cross-temporal calibration.** An agent calibrated on 2020-2023 data should remain calibrated on 2024 data. If calibration degrades sharply on the test set, the agent is overfitting to the training period's distribution of claims, not learning transferable judgment.
- **Comparison to prediction market baselines.** Where ML prediction markets exist (Metaculus questions about AI capabilities, ML benchmarks), compare the simulated market's credence trajectories to actual prediction market prices. They should correlate. If the simulation's credence paths look nothing like how real humans updated on the same evidence, the simulation is unrealistic.

### Evidence pipeline validation

This is the most important faithfulness check because it's where the most bias can enter:

- **Claim extraction audit.** Manually review 200 randomly sampled conjecture extractions. Score: does the extracted claim faithfully represent the paper's actual claim? Does the direction label (supports/contradicts) correctly capture the relationship? Target: >85% accuracy on both.
- **Strength signal validation.** Compare the citation-velocity strength signal against expert judgments of evidence quality for 100 paper-conjecture pairs. Target: Spearman correlation > 0.5.
- **Dependency graph audit.** For the 50 highest-degree conjectures, manually verify the top-5 dependencies. Are these real logical dependencies or citation artifacts? Target: >70% precision.

### Transfer test (the real test)

The ultimate faithfulness check: does training on the simulated market improve performance on tasks outside the simulation?

- **Held-out domain transfer.** Train on ML conjectures, test on a small set of biology or physics conjectures constructed the same way. If the agent's calibration and evidence discrimination transfer, the skills are general, not ML-specific.
- **Live prediction.** Take 20 open ML questions (from Metaculus, or questions we write ourselves about 2025 developments). Have the trained agent predict credences. Revisit in 6-12 months. Compare to zero-shot LLM predictions on the same questions. If the trained agent is measurably better calibrated on live questions, the simulation trained something real.
- **Expert Turing test.** Show ML researchers the agent's trading rationale (why it bought YES or NO on a conjecture) alongside a human expert's rationale. Can they distinguish them? If the agent's reasoning is indistinguishable from a knowledgeable human's, the simulation is producing genuine scientific judgment, not pattern matching.

### Known biases and limitations

These are the ways the simulation is unfaithful, documented so users can account for them:

- **Survivorship bias.** We reconstruct conjectures from papers that were published and indexed. Claims that were discussed informally but never published are invisible. The simulation overrepresents "important" conjectures and underrepresents the long tail of mundane claims that constitute most of real scientific discourse.
- **Hindsight in labeling.** The direction labels (supports/contradicts) are assigned with knowledge of eventual outcomes. A paper that seemed supportive in 2018 but was later shown to be flawed will be labeled "contradicts" in our dataset. This is conservative — it makes the benchmark harder — but it's not how information arrived in real time.
- **No social dynamics.** Real science involves reputation, funding pressure, advisor relationships, and conference politics. The simulation treats evidence as text, ignoring who wrote it and why. An agent trained here won't learn to discount a paper from a lab with a track record of irreproducible results (unless that signal shows up in the citation velocity).
- **Fixed evidence order.** The simulation replays evidence in the order it was actually published. In a real market, agents' trades would influence which experiments get run next (path dependence). The simulation is a replay, not a counterfactual simulation. The agent can't "cause" new evidence to appear by taking a position.

### Faithfulness score

We propose a composite faithfulness score for the simulation itself (not for agents):

```
Faithfulness = 0.3 * credence_convergence_rate
             + 0.2 * entropy_monotonicity_rate
             + 0.2 * claim_extraction_accuracy
             + 0.2 * strength_signal_correlation
             + 0.1 * dependency_precision
```

Target: > 0.75. Below 0.6 means the simulation needs pipeline fixes before benchmark results are meaningful.

---

## Implementation Plan

### Phase 1: Data pipeline (weeks 1-2)

1. Pull ML papers 2020-2024 from Semantic Scholar API (metadata + abstracts)
2. Extract Level 1 conjectures from Papers With Code SOTA tables
3. Extract Level 2 conjectures from abstracts using LLM
4. Build citation-based dependency graph
5. Compute citation velocity strength signals
6. Construct evidence timelines per conjecture

### Phase 2: Environment (weeks 2-4)

1. Implement `ScienceMarketEnv` with Gymnasium API
2. Implement entropy-based pricing and credence updates
3. Implement portfolio tracking and reward computation
4. Build episode sampling (train/val/test splits by time)
5. Write baseline agents (random, majority, citation, zero-shot LLM)

### Phase 3: Benchmark validation (weeks 4-5)

1. Run faithfulness sanity checks
2. Audit claim extraction (200 samples)
3. Audit dependency graph (50 high-degree nodes)
4. Run all baseline agents, publish initial leaderboard
5. Compute faithfulness score

### Phase 4: RL training (weeks 5-7)

1. Train RL agents (PPO, DPO) on the environment
2. Compare RL-trained agents to zero-shot LLM baseline
3. Ablate: does market structure (entropy pricing, portfolio) help vs. just reading papers?
4. Run transfer test on held-out domain
5. Write up results
