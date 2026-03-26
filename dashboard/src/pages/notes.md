---
publish: true
unlisted: true
title: "Science Market"
date: 2026-03-22
description: "A position paper on scientific discovery, veracity, and market mechanisms in the age of AI."
---
  
Notes to add
- Federated vs centralized event bus
  - do all agents actually need to be on the same event bus as all other agents? or should some event buses be federated, aka scoped locally? a la github.
- right now, there is no steam engine. ai research is still pre-paradigmatic.
- how do people actually know that research is true? Are the quantifiable signals for veracity? how will we know that there are answers?
  - difference between "answers" and "improvements" and "x" in the kinds of results of research.
  - how do you find mistakes?
  - research and code may seem different, but actually they're not. 
    - both rely on empiricism - tests, artifacts, etc.
  - modes of verification
    - empiricism (observation of evidence that something is true)
      - "if you can't tie a claim to an observation/experience, don't trust it" - Hume
      - strengths
        - anchors belief
        - enables falsification
        - scales with instrumentation
      - limitations
        - induction / failure to generalize
        - overfitting
        - data without theory is limited
    - theory: structure, compression, prediction
      - what is it
        - "internal models to organize / interpret observations" - Descartes
        - a compression/topology of data/observations
        - a predictive engine
        - a causal story
      - strengths
        - generalization
        - conterfactuals
        - guides what data to collect
      - weaknesses
        - detached abstraction
        - underdetermination: multiple theories explain the same thing
        - motivated reasoning
- how different is research from engineering?
  - in engineering, the end goal is usually the solution to a problem. in mathematics research, the problem is usually a means to identify new tools to solve the problem, ie discovering the process is the goal. Is this true in other domains of research?
  - math historically has not been experiment-based. math research in the age of AI will be less about solving a problem, and more about using search/proof experiments to identify strategies for solving problems. [tao]
- how should research agents do work that is well structured / aligned?
  - pedagogical viability / coherence: how well do the results lend themselves to being interpreted by humans? interpretability. 
  - minimizing consensus building friction: how will you create artifacts that minimize the amount of effort required for humans to accept the veracity? is there a difference in how humans verify vs agents verify?
  - agents either succeed or fail, but don't yet create partial stages. [tao]
  - AIs excel at breadth, and humans excel at depth. [tao]
  - bayesianism: theory → prediction → experiment → data → update theory → ...
    (a) Empirical adequacy
        •	Does it match observations?
        •	Can it predict new data?

    (b) Falsifiability
        •	Could it, in principle, be proven wrong?
    (Popper’s criterion)

    (c) Predictive power
        •	Does it generalize beyond the training set?

    (d) Parsimony (Occam’s Razor)
        •	Does it explain with minimal assumptions?

    (e) Coherence
        •	Does it fit with other well-supported theories?

    (f) Causal robustness
        •	Does it survive intervention (not just correlation)?
- on novel theory artifacts: continuous laws (prime number theorem) vs discrete topologies (complexity classes)
- building science historiography as a rubric for useful science

So going a bit more on the position paper style.

# Position

## Current problems in science

### Agents are not good enough yet
- technological limitations of agents
  - lack of continual learning: after an AI writes a proof, if you spin up a new session with it, it doesn't remember how it solved the problem before.

### Reproduceability crisis
- veracity challenges: it's not obvious how individuals should evaluate/verify research results. Very related to the reproducibility crisis.

### Peer review is collapsing
- peer review collapsing under AI throughput: as humans generate and publish more research, peer review can't keep up, so science stalls.

### Publication bias against null results and "half-truths"
- because reviewers are limited, rarely do null findings get published and disseminated.

Much of this has to do with the fact that science is participatory: research benefits from having people review it, challenge components, reproduce it.

## Discovery Microeconomics

Consider these challenges in science from the lens of microeconomics: in what ways does the current "marketplace of ideas" struggle to support rapid acceleration of individual scientist production, the rapid increase in participants, and in what ways can we change the marketplace to support both, while maintaining the quality of the science as a whole? By applying this lens, we can use principles from microeconomic theory to guide how humans and autonomous agent scientists can modify their behaviors to improve the marketplace, and ways we can restructure the marketplace to work better for all participants.

Concretely, let's define a **scientist** as a firm/participant activity to create new results and add them to the marketplace of ideas, and **science** as the marketplace itself. scientists participate in science by exchanging, evaluating, and incorporating results of others. Some common metrics to measure scientists are productivity (results per time unit, e.g. hours), citation count, and h-index. Some metrics to measure science (the marketplace) include the number of scientists, the result adoption velocity (how quickly results are integrated by other scientists), and the idea exchange network connectivity (how much scientists exchange with each other).

If we experience a massive growth in the number of scientists (due to lower barriers to entry by AI) and the speed of research production (due to increased throughput by AI), it will lead to real changes for science and scientists that the existing marketplace can't support.

### Price discovery as result credence

In markets, price discovery occurs when a transaction between a buyer and seller is broadcasted to the market. As new information about the underyling value of the asset becomes available, the price changes. 

Determing the veracity of results and determing the value of goods are both discovery procedures under uncertainty typically driven by distributed consensus mechanisms (with some important exceptions). Today, scientists rely on paper citation count, scientist reputation, peer review, and reproductions as heuristics for result veracity. These are slow signals to generate, as its bounded by scientists reviewing results and producing new papers that cite that result, which itself is bottlenecked by the time to produce results and the number of scientists working in that domain. 

This is deeply related to prediction markets, where participants trade contracts whose payoff depends on a future event. A "conjecture market" could work similarly, where all scientific artifacts are rephrased into conjectures with predictive capacity and thus falsifiable statements. Then, results which depend on prior results have contracts that can be priced relative to the contracts of the dependencies. Another way to think about this: most publications don't come with confidence intervals, but even results with low confidence can still be useful and incorporated. If confidence in the veracity of a result has superlinear costs (which seems true), then science will be expensive.

Consider it from a Bayesian perspective: let's say $A$ is my result being "true", and $P(A)$ is the probability it is, which we'll say is 80%. If a new conjecture $B$ is made that depends on $A$ alongside other conjectures $X$, and through their own independent work that they believe that $B$ is true, also at 80%, then $P(A \mid X) = 0.8$ and $ P(B \mid A, X) = 0.8$.

The posterior would be

$$
P(A \mid B, X) = \frac{P(B \mid A, X),P(A \mid X)}{P(B \mid X)}
$$

Then
$$
P(B \mid X) = P(B \mid A, X)P(A \mid X) + P(B \mid \neg A, X)P(\neg A \mid X).
$$


Since $P(A \mid X)=0.8$, we have $P(\neg A \mid X)=0.2$, so

$$
P(B \mid X) = 0.8 \cdot 0.8 + P(B \mid \neg A, X)\cdot 0.2.
$$

So

$$
P(A \mid B, X) = \frac{0.64}{0.64 + 0.2 P(B \mid \neg A, X)}.
$$

So if $B$ is likely to be true regardless of whether $A$ is true, then our posterior on $A$ shouldn't change. But if $B$ heavily depends on $A$ being true, and we observe $B$, then $A$ should be upweighted.

The marketplace aims to reward participants based on the usefulness of results scientists create. 

$$
  \text{value of contribution} \propto \text{marginal improvement in calibrated future prediction}
$$

Consider a scientist who has as new result $B$ that supports $A$. In the conjecture market, they would go and buy $A$ before they published their result $B$. This action represents an implicit belief in $P(B | \neg A, X)$ being small, as the scientist wouldn't buy $A$ if it were unrelated to $B$.

Note that scientists don't ever personally report their confidence or probabilities, that these are instead downstream of the price of the contracts on $A$ and $B$.  

### Testability as epistemic liquidity

Liquidity matters: in deep markets, prices adjust smoothly; in thin markets, small trades can move the price.

A liquid market lets information move into prices quickly because participation is easy. An epistemically liquid field lets claims get evaluated quickly because tools, benchmarks, data, and reviewer bandwidth are available. Illiquid knowledge systems have lots of claims but few decisive tests. The ability to rapidly reproduce experimental results or quickly verify proofs improves the liquidity of the marketplace.

### Market design

Foundational principles of market design are that markets should be thick, deep, uncongested, and safe. "Thick” means enough participants and opportunities show up in one place to make good matches possible. "Deep" means that the market has a lot of resting buy/sell interest near the current price, so that large orders can trade with limited price impact. “Uncongested” means the system gives participants enough structure, time, and mechanism to evaluate options instead of getting overwhelmed or forced into premature decisions. “Safe” means participants can act on their true preferences without excessive gaming, adverse selection, or fear that honest participation will punish them.

We are pre-paradigmatic on AI science research. As a result, we propose the need to formalize the contours of what is needed to do science in an era where humans are the bottleneck. 

Core problems we highlight for agentic science to scale macro-scientifically:

- human bottlenecks
- an abscence of ontology/taxonomy of research artifacts
  - results: 
    - currently extremely diverse: "A new proof for x", or "A new method for x", or "A new measurement/observation of x", or "A new prediction for x". 
    - different fields of science aim to produce different results, and even subfields can vary widely. For example, the prime number theorem is a statistical observation, not a proof. 
  - evidence: 
    - code, experiments, data.
  - why it matters
- a dearth of negative result reporting.
  -  

## Appendix
## Problems I anticipate
- Supply and demand: prices coordinate buyers and sellers.
- Marginal thinking: decisions are made at the margin, not in all-or-nothing terms.
- Opportunity cost: choosing one thing means giving up another.
- Elasticity: how responsive behavior is to changes in price, income, or policy.
- Market structure: competition, monopoly, oligopoly, bargaining.
- Externalities: actions affecting others outside the transaction, like pollution.
- Information problems: adverse selection, moral hazard, signaling.
- Game theory / mechanism design: strategic interaction and rule design.

Then you should not build the system around binary resolution (like a prediction market). You should build it around continual forecast improvement.

If conjectures never fully resolve, the right objective is not “who was right in the end?” but “who improved the community’s beliefs in a way that later evidence validated better
than the alternative?”

That pushes you toward a recursive scoring system:

- Each conjecture (A) has a live market state (p_t(A)).
- People can submit updates, evidence, decompositions, replications, or downstream consequences.
- They are rewarded based on whether their update improves future predictive performance relative to the prior state.

Instead of final resolution, use rolling evaluation:

- Compare $p_t(A)$ to later states $p_{t+\Delta}(A)$, downstream predictions, and observable consequences.
- Reward updates that move (A) in directions later supported by further evidence.
- Penalize updates that move (A) in directions later reversed.

A simple form is a market on log-likelihood improvement:
$$
\text{reward}(i) \approx \log p_{t+\Delta}^{(ref)}(A) - \log p_t^{(ref)}(A)
$$
with the sign depending on the direction of the user’s move. More generally, reward agent (i) by how much their update improves predictive accuracy on future observable
variables.

So the object being scored should often not be (A) directly, but the set of predictions implied by (A).

That is the key move:

- Treat a conjecture as valuable insofar as it compresses and predicts future observations.
- Reward people for improving that predictive package.

In practice, this means each conjecture needs:

- A current confidence (p(A)).
- A set of attached predictive commitments.
- Links to downstream conjectures and empirical observables.
- A rolling scoring horizon.

Then incentives work like this:

- If I strengthen (A), I am implicitly backing the predictions (A) makes.
- If those predictions perform well relative to alternatives, I earn continuing reward.
- If later work undermines those predictions, my earlier reward decays or gets clawed back.

So compensation becomes provisional, not final.

A useful mental model is:

- publications become updates,
- citations become dependency edges,
- replications become evidence shocks,
- and “truth” is replaced by predictive track record over time.

This avoids the impossible requirement of ultimate resolution, but it creates two hard problems.

First, benchmark choice:
You need to decide what counts as “later validation.” Usually this means:

- later market states,
- later replications,
- predictive success on held-out data,
- or success of descendant conjectures.

Second, reflexivity:
If later market states determine rewards, you can get herding or manipulation. So you need some combination of:

- delayed scoring,
- independent audit markets,
- multiple evaluators,
- and rewards based partly on real-world predictive outcomes, not just consensus.

So the mechanism becomes less like a standard prediction market and more like a continuously updated epistemic capital system.

The core principle is:

$$
\text{value of contribution} \propto \text{marginal improvement in calibrated future prediction}
$$

not

$$
\text{value of contribution} \propto \text{eventual binary correctness.}
$$

If you want a concise framing for your writeup, it could be:

> Scientific claims rarely admit final resolution; they remain revisable in light of new evidence. Therefore a conjecture market should not rely on binary settlement. Instead, it
> should reward contributors for marginal improvements in the predictive performance and calibration of the conjecture network over time. Compensation is thus provisional and
> continuously updated, reflecting whether a contribution is borne out by later evidence, descendant conjectures, and empirical predictions.

1. Rolling Scores

Each conjecture (A) has a live credence $p_t(A)$, but contributors are not paid on final resolution. They are paid on how much their update improves later predictive performance.

Let agent (i) move the market from $p_t$ to $p_t'$. Attach to (A) a set of observable implications $Y_A = {y_1,\dots,y_n}$, each with a forecast horizon. Then score the update
against later observations or later validated forecasts:

$$
\text{Reward}_i(A) =

\sum_{y \in Y_A} w_y 
\left[
  S\left(
    p_{t\,\text{after }i}(y)\,\,o_y
    \right) 
    - S\left(
      p_{t\,\text{before }i}(y)\,\,o_y
      \right)
\right]
$$

where (S) is a proper scoring rule, such as log score,

$$
S(p,1)=\log p,\qquad S(p,0)=\log(1-p).
$$

If outcomes are not binary, use the appropriate probabilistic score. The point is that contributors are rewarded for improving forecasts of consequences, not for claiming
certainty about (A) itself.

To avoid immediate payout on noisy updates, use delayed vesting:

$$
\text{Payout}_i(t) = \sum{\Delta \in H} \alpha_\Delta ,\text{Reward}_i^{(t+\Delta)}
$$

with horizons (H) like 3 months, 1 year, 3 years. This makes rewards provisional and continuously revised.

2. Dependency Graph Propagation

Represent science as a directed acyclic graph, or approximately so, with conjectures as nodes and dependency weights on edges.

If (B) depends on (A) and other conjectures (X), then the market stores a structural relation such as

$$
P(B) = f!\big(P(A), P(X_1), \dots, P(X_k)\big).
$$

A local approximation is often enough:

$$
\Delta \sigma P(B) \approx \beta_{BA},\Delta \sigma P(A) + \sum_j \beta_{Bj},\Delta \sigma P(X_j).
$$

Then when someone updates (B), some of that information can flow back to (A), but only in proportion to how diagnostic (B) is of (A). In Bayesian terms, the relevant quantity is
still the likelihood ratio:

$$
\frac{P(B \mid A, X)}{P(B \mid \neg A, X)}.
$$

So the system should not let “belief in (B)” automatically increase (A). It should increase (A) only when (B) would have been substantially less expected if (A) were false.

A useful decomposition is:

$$
\text{Information contributed to }A\text{ by update on }B
\propto
\log \frac{P(B \mid A,X)}{P(B \mid \neg A,X)}.
$$

That lets you propagate evidence upstream while avoiding naive double-counting.

Operationally:

- Each edge carries an estimated diagnostic strength.
- Updates propagate through the graph with attenuation.
- Only residual information is rewarded.
- If (B) was derived mainly by assuming (A), the back-propagation to (A) is near zero.

3. Anti-Gaming Rules

This system will be easy to game unless constraints are explicit.

Use these rules:

- Require operationalized conjectures.
  Every node must generate at least some observable predictions, benchmark tasks, or compression claims. Vague claims cannot earn much.
- Separate update rights from payout timing.
  People can move markets anytime, but rewards vest only after later evidence arrives.
- Use stake/slashing.
  Large updates require collateral. If the update later degrades predictive performance, some stake is lost.
- Reward marginal contribution only.
  Payment is based on improvement over the previous state, not absolute ownership of an idea.
- Use path-independent accounting where possible.
  A Shapley-style approximation can split credit across overlapping contributors, though exact Shapley values may be too expensive. A sequential log-score decomposition is the
  practical substitute.
- Limit reflexive scoring.
  Do not score updates only by later market opinion. Use external anchors:
  replications, benchmark predictions, held-out data, intervention outcomes, or downstream conjectures that themselves have empirical hooks.
- Penalize circular support.
  If (A) supports (B) and (B) is then cited as evidence for (A), the system should discount that loop unless new empirical content entered.
- Track calibration by participant.
  A user’s trust weight should depend on long-run forecast calibration, not status.
- Make manipulation expensive.
  Use automated market makers or bounded liquidity in thin nodes, audit logs, and delayed settlement of large rewards.
- Distinguish evidence from derivation.
  A theorem, model, or paper that assumes (A) is not independent evidence for (A) unless it yields new successful predictions.

A concise mechanism is:

1. Each conjecture is a node with a live probability and a bundle of predictive commitments.
2. Contributors can update nodes by staking capital and attaching evidence.
3. Updates propagate through a dependency graph using estimated diagnostic weights.
4. Rewards are paid from rolling proper-score improvement on future observations and validated descendant predictions.
5. Rewards vest gradually and can be clawed back if later evidence reverses the contribution.

A compact formal objective is:

$$  \text{Contribution}_i

\sum_{t,;y}
\alpha_t w_y
\left[
S!\left(p^{(i)}_{t}(y), o_y\right)

S!\left(p^{(-i)}_{t}(y), o_y\right)
\right]
$$

where (p^{(i)}_t(y)) is the forecast path with agent (i)'s update and (p^{(-i)}_t(y)) is the counterfactual without it. In practice you approximate the counterfactual
sequentially.

The underlying philosophy is:

- no conjecture ever fully resolves,
- so value must come from improving future prediction,
- and scientific credit should accumulate from marginal predictive usefulness, not publication finality.


> okay, what if you pay to submit conjectures or beliefs (staking), and your residuals are proportionate to how much you initially stake? and we have some geometric-decay for
attribution where if you have a belief, and someone adds evidence, so they buy shares in A, and they pay to submit their evidence, and when they do the owners of A get fractions
of that submission, and then when the evidence shifts for A, the adder of the evidence gets minted access to new shares of A (ie the shares of A get diluted, since adding good
evidence is like gaining ownership in A)?


Yes. That is a coherent mechanism.

What you are describing is basically a dynamic cap table for conjectures: early stakers are like founders, later evidence contributors are like investors or employees who earn
diluted ownership by increasing the conjecture’s predictive value.

The main correction is this: residuals should be proportional to current diluted ownership, not permanent initial stake. Initial stake should only determine the initial share
allocation.

A clean version looks like this.

1. For each conjecture (A), keep two separate objects:
$$
    p_t(A) \quad \text{current belief / forecast}
$$ 
and
$$
    N_t(A);\,\theta_i^t(A) \quad \text{share supply and ownership fractions.}
$$

This separation matters. If the same token both sets belief and captures royalties, the system is too easy to game.

2. When (A) is created, the founder stakes ($s_0$) and receives founder shares:
$$
N_0(A) = \phi(s_0).
$$
3. When contributor (j) submits evidence on (A), they pay a stake ($s_j$) and propose an update
$$
p_t(A) \to p_t'(A).
$$

Their payment is split:
$$
s_j = F_j^{\text{holders}} + F_j^{\text{ancestors}} + F_j^{\text{treasury}},
$$
for example
$$
F_j^{\text{holders}} = \alpha s_j,\quad
F_j^{\text{ancestors}} = \beta s_j,\quad
F_j^{\text{treasury}} = (1-\alpha-\beta)s_j.
$$

Current holders of (A) split $F_j^{\text{holders}}$ pro rata by ownership. So yes, new evidence pays incumbents.

4. The contributor also earns provisional new shares if the evidence is good. Do not mint them just because price moved. Mint them based on validated information gain:
$$
m_j^{\text{prov}}(A) = \kappa ,\widehat{\Delta S}_j(A),
$$
where $\widehat{\Delta S}_j(A)$ is the expected future score improvement from that evidence.

Then only a fraction vests later:
$$
m_j^{\text{vest}}(A) = v_j, m_j^{\text{prov}}(A), \qquad 0 \le v_j \le 1.
$$

If the evidence ages well, (v_j) is high. If it gets reversed, (v_j) is low and some stake can be slashed.

5. Once shares vest, existing holders are diluted:
$$
\theta_i^{t+1}(A)
=
\frac{n_i^t(A) + m_i^{\text{vest}}(A)}
{N_t(A) + \sum_k m_k^{\text{vest}}(A)}.
$$

That is the core intuition: good evidence increases the value of (A), and the contributor is paid in ownership of that improved asset.

For your geometric-decay attribution, if (B) depends on (A), then a submission fee on (B) can flow upstream to (A) and its ancestors. A simple rule is:
$$
w_u(B) = \frac{\lambda^{d(u,B)}}{\sum_{v \in \mathrm{Anc}(B)} \lambda^{d(v,B)}},
\qquad 0<\lambda<1,
$$
where $d(u,B)$ is graph distance. Then ancestor (u) receives
$$
F_u(B) = w_u(B)\,F^{\text{ancestors}}(B).
$$

So yes: upstream conjectures can earn royalties from downstream use.

What this buys you:

- Founders of useful conjectures earn ongoing residuals.
- New evidence contributors can buy in and dilute incumbents.
- Downstream work pays upstream dependencies.
- Knowledge becomes something like equity plus royalties.

What will break if you do not handle it carefully:

- If minting happens on immediate price movement, people will manipulate thin markets.
- If there is no delayed vesting/clawback, noise gets capitalized permanently.
- If there is no outside revenue, then old holders are just being paid by new entrants.
- If residuals depend on initial stake forever, later contributors never truly own anything.

So the right version is:

- forecast state for belief,
- equity shares for residual cashflows,
- staking to submit evidence,
- delayed minting based on later predictive validation,
- pro rata dilution,
- and geometric upstream royalties.

This is less a standard prediction market and more a knowledge royalty protocol.

If you want, I can write this as a compact formal mechanism with named components and one worked example.

# A compact version is a conjecture equity market. The key separation is:

$$
p_t(A) \neq \text{ownership of }A
$$

(p_t(A)) is the live credence in conjecture (A). Ownership is a separate cap table that earns future residuals.

Mechanism

1. Each conjecture (A) has:
$$
p_t(A), \quad N_t(A), \quad n_i^t(A), \quad Y_A, \quad Anc(A)
$$

where (N_t(A)) is total shares, (n_i^t(A)) is user (i)'s shares, (Y_A) is a set of predictive consequences, and (Anc(A)) are upstream dependencies.

2. Creation:
A founder stakes (s_0) to create (A), receives founder shares
$$
N_0(A)=\phi(s_0),
$$
and sets or seeds the initial credence (p_0(A)).

3. Evidence submission:
A contributor (j) submits evidence (e_j) with stake (s_j), proposing
$$
p_t(A)\to p_t'(A).
$$
To make manipulation expensive, bound the move by stake:
$$
\left|\operatorname{logit} p_t'(A)-\operatorname{logit} p_t(A)\right|
\le c\log(1+s_j).
$$
4. Submission fee split:
$$
s_j=\alpha s_j+\beta s_j+\gamma s_j,\qquad \alpha+\beta+\gamma=1.
$$
The (\alpha) piece goes to current holders of (A):
$$
r_{ij}^{\text{direct}}(A)=\alpha s_j\frac{n_i^t(A)}{N_t(A)}.
$$
The (\beta) piece goes upstream with geometric decay. For ancestor (u\in Anc(A)),
$$
\tilde r_{u\leftarrow A,j}
=
\beta s_j,
\frac{\lambda^{d(u,A)}}{\sum_{v\in Anc(A)}\lambda^{d(v,A)}},
\qquad 0<\lambda<1,
$$
and holders of (u) split that amount pro rata. The (\gamma) piece goes to treasury / validators / market maintenance.
5. Rolling validation:
The evidence is not judged by final truth, but by later predictive performance. At future horizons (H={\tau_1,\tau_2,\dots}),
$$
\Delta S_j(A)
=
\sum_{\tau\in H}\omega_\tau
\sum_{y\in Y_A} w_y
\Big(
S(q_{j,\tau}(y),o_{y,\tau})


S(q_{0,\tau}(y),o_{y,\tau})
\Big),
$$
where (S) is a proper scoring rule, (q_{j,\tau}) is the forecast path after (j)'s evidence, and (q_{0,\tau}) is the path before it. For binary events, one choice is
$$
S(p,1)=\log p,\qquad S(p,0)=\log(1-p).
$$

6. Minting and dilution:
Good evidence mints new shares:
$$
m_j(A)=\kappa\max(\Delta S_j(A),0).
$$
Then
$$
N_{t+1}(A)=N_t(A)+m_j(A),\qquad
n_j^{t+1}(A)=n_j^t(A)+m_j(A).
$$
So existing holders are diluted when someone adds genuinely valuable evidence. If (\Delta S_j(A)<0), no shares vest and some stake can be slashed:
$$
\text{slash}_j=\mu\max(-\Delta S_j(A),0).
$$
7. Residuals:
A conjecture earns cash flow from new submissions, downstream usage, and any external demand for its forecasts:
$$
R_t(A)
=
\sum_{j\in Sub(A,t)}\alpha s_j


\sum_{B\succ A}\sum_{j\in Sub(B,t)}\tilde r_{A\leftarrow B,j}
+
Q_t(A).
$$
Each holder gets
$$
\pi_i^t(A)=\frac{n_i^t(A)}{N_t(A)}R_t(A).
$$

# Example

Alice creates conjecture (A) with stake (\$100). Let
$$
\phi(s)=10s,
$$
so she receives (1000) shares. Initial credence is $p_0(A)=0.60$.

Bob later submits evidence on (A) with stake (\$20), shifting credence to (0.70). Let
$$
(\alpha,\beta,\gamma)=(0.5,0.2,0.3).
$$
So Bob's submission immediately pays (\$10) to current holders of (A). Since Alice is the only holder, she receives the full (\$10).

Over the next year, Bob's evidence improves (A)'s rolling score by
$$
\Delta S_B(A)=0.08.
$$
If $\kappa=1000$, Bob is minted
$$
m_B(A)=1000\cdot 0.08=80
$$
new shares.

The cap table becomes:
$$
N(A)=1080,\qquad
\theta_{\text{Alice}}(A)=\frac{1000}{1080}\approx 92.6\%,\qquad
\theta_{\text{Bob}}(A)=\frac{80}{1080}\approx 7.4%.
$$

Now Carol creates downstream conjecture (B) that depends on (A), staking (\$50). If (A) is the only ancestor of (B), then (A) receives the full upstream royalty:
$$
\tilde r_{A\leftarrow B}=0.2\cdot 50=\$10.
$$
That (\$10) is split by current ownership of (A):
$$
\text{Alice gets } \$9.26,\qquad
\text{Bob gets } \$0.74.
$$

So Bob has converted knowledge into money in two ways:

- he changed the credence of (A),
- and because that change later proved predictively useful, he gained equity in (A), which now earns future residuals.

### Interpretation

This matches your intuition pretty closely:

- Initial staking sets the initial cap table.
- New evidence pays incumbents immediately.
- Good evidence later dilutes incumbents and gives the contributor ownership.
- Downstream use sends royalties upstream with geometric decay.
- No conjecture needs final resolution; vesting is based on rolling predictive success.

The main non-negotiable design choice is that credence and ownership must stay separate. Otherwise moving belief and extracting rent collapse into the same token and the whole
thing becomes too easy to game.

If you want, I can turn this into a short essay-style note with cleaner prose and a name for the mechanism.

# Discussion: Does veracity even matter in science?

A common objection: people publish preprints to arXiv without peer review, and science seems to work fine. So does veracity actually matter?

## The short answer: veracity matters, but verification is what's broken

The fact that arXiv exists and is enormously useful proves something important—but not what the objection claims. ArXiv shows that *dissemination without verification* is valuable. It does not show that *verification itself* is unnecessary.

What actually happens with arXiv preprints:

1. **Preprints are not the end state.** Most impactful arXiv papers eventually go through peer review, get cited, get reproduced, or get quietly ignored. The arXiv is a *staging area*, not a replacement for evaluation. The community still performs verification—it just does it informally, slowly, and unevenly.

2. **Readers apply their own verification heuristics.** When a researcher reads an arXiv paper, they are not treating all claims as equally credible. They check who wrote it, what institution they're at, whether the proofs look right, whether the experiments are reproducible. This is distributed, informal peer review—it's just not formalized or recorded anywhere.

3. **The failures are real and costly.** Papers that turned out to be wrong have caused real damage: wasted follow-up research, retracted clinical findings, misallocated funding. The reproducibility crisis is not hypothetical—it's estimated that $28 billion per year is spent on irreproducible preclinical research in the US alone (Freedman et al., 2015). ArXiv's lack of gatekeeping makes this worse, not better, when the informal verification layer fails.

## What arXiv actually demonstrates

ArXiv demonstrates that the *speed of dissemination* and the *speed of verification* are separate problems, and that solving the first without the second is already valuable. But it also demonstrates the cost of leaving the second unsolved:

- **No confidence signal.** A preprint on arXiv has no attached credence. Is this a breakthrough or noise? Readers must evaluate from scratch every time. There is no mechanism for aggregating the community's collective assessment.
- **No incentive for verification.** Nobody gets credit for reading a preprint carefully and saying "this is correct" or "this is wrong." The incentive structure rewards *producing* new work, not *evaluating* existing work.
- **Thin epistemic liquidity.** Many arXiv papers sit in a kind of limbo—cited by some, ignored by others, never definitively evaluated. The information exists but hasn't been priced.

This is precisely the market failure the conjecture market addresses. The problem is not that scientists don't care about truth—they do. The problem is that the *mechanisms for establishing and communicating credence* are too slow, too informal, and too poorly incentivized.

## Veracity as a spectrum, not a binary

The deeper issue is that the question "does veracity matter?" frames truth as binary: a result is either true or false, verified or unverified. But science doesn't work that way. Most results live on a spectrum of credence:

- A new theoretical prediction with no experimental support might warrant $p = 0.3$.
- A well-reproduced experimental result might warrant $p = 0.95$.
- A result that contradicts established theory but has strong data might warrant $p = 0.5$ pending further investigation.

The current system has no way to express this. A paper is either published or not. It's either peer-reviewed or not. These are coarse binary signals in a world that needs continuous ones.

ArXiv preprints are implicitly low-credence publications—everyone knows they haven't been reviewed. But "low credence" is not "no credence needed." The value of a preprint depends entirely on how much you should believe it, and right now there's no efficient mechanism for aggregating that belief.

## The conjecture market reframe

In the conjecture market framework, arXiv's role becomes clear: it is a mechanism for *submitting evidence to the market*. The missing piece is the market itself—the infrastructure that:

1. Attaches a live credence to each claim
2. Rewards participants for improving that credence (in either direction)
3. Propagates evidence through the dependency graph of scientific knowledge
4. Makes verification a compensated activity, not a thankless chore

Veracity doesn't stop mattering because we stopped formally checking it. It means we're flying blind—disseminating claims without pricing them. The conjecture market is an attempt to build the pricing mechanism that arXiv lacks.

## Summary

| Aspect | Current state (arXiv + peer review) | Conjecture market |
|--------|--------------------------------------|-------------------|
| Dissemination speed | Fast (arXiv) | Fast |
| Verification speed | Slow (months to years) | Continuous |
| Credence signal | Binary (reviewed/not) | Continuous price |
| Incentive to verify | Weak (no credit) | Strong (stake + residuals) |
| Null results | Rarely published | Equally valued |
| Partial confidence | Not expressible | Native to the system |

The question is not whether veracity matters—it's whether we can build systems that establish it faster and more efficiently than the ones we have.
