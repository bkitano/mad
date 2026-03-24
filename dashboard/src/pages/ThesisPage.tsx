import { useState, useEffect } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

interface Section {
  id: string
  title: string
  level: number
}

const sections: Section[] = [
  { id: 'introduction', title: 'Introduction', level: 1 },
  { id: 'motivation', title: 'Motivation', level: 2 },
  { id: 'core-hypothesis', title: 'Core Hypothesis', level: 2 },
  { id: 'architecture-search', title: 'Architecture Search with AI', level: 1 },
  { id: 'search-space', title: 'The Search Space', level: 2 },
  { id: 'evaluation-strategy', title: 'Evaluation Strategy', level: 2 },
  { id: 'attention-mechanisms', title: 'Attention Mechanisms', level: 1 },
  { id: 'softmax-attention', title: 'Softmax Attention', level: 2 },
  { id: 'linear-attention', title: 'Linear Attention', level: 2 },
  { id: 'kernel-methods', title: 'Kernel Methods', level: 2 },
  { id: 'experimental-results', title: 'Experimental Results', level: 1 },
  { id: 'scaling-laws', title: 'Scaling Laws', level: 2 },
  { id: 'efficiency-benchmarks', title: 'Efficiency Benchmarks', level: 2 },
  { id: 'future-directions', title: 'Future Directions', level: 1 },
]

export default function ThesisPage() {
  const [activeSection, setActiveSection] = useState('introduction')

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id)
          }
        }
      },
      { rootMargin: '-80px 0px -60% 0px' }
    )

    sections.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })

    return () => observer.disconnect()
  }, [])

  const scrollTo = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <div className="thesis-page min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-6xl mx-auto px-6 py-12 flex gap-12">
        {/* Side Table of Contents */}
        <nav className="hidden lg:block w-64 flex-shrink-0">
          <div className="sticky top-24">
            <h3
              className="text-xs font-semibold uppercase tracking-widest mb-6"
              style={{ color: 'var(--ink-muted)', fontFamily: 'var(--font-display)' }}
            >
              Table of Contents
            </h3>
            <ul className="space-y-1">
              {sections.map(({ id, title, level }) => (
                <li key={id}>
                  <button
                    onClick={() => scrollTo(id)}
                    className="block w-full text-left transition-colors duration-150 rounded px-2 py-1.5 text-sm"
                    style={{
                      paddingLeft: level === 2 ? '1.25rem' : '0.5rem',
                      fontFamily: 'var(--font-display)',
                      fontWeight: level === 1 ? 500 : 400,
                      fontSize: level === 1 ? '0.875rem' : '0.8125rem',
                      color: activeSection === id ? 'var(--accent-strong)' : 'var(--ink-muted)',
                      backgroundColor: activeSection === id ? 'var(--accent-soft)' : 'transparent',
                    }}
                  >
                    {title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        {/* Main Content */}
        <article className="flex-1 min-w-0">
          <header className="mb-16">
            <h1
              className="text-4xl font-bold mb-4 leading-tight"
              style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
            >
              Notes on AI-Driven Architecture Search
            </h1>
            <p
              className="text-lg leading-relaxed"
              style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
            >
              A collection of working notes on using AI agents to explore the space of neural
              network architectures, with a focus on efficient attention mechanisms and
              automated experimentation.
            </p>
            <div
              className="mt-4 text-sm"
              style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}
            >
              Last updated: March 2026 &middot; Working draft
            </div>
          </header>

          {/* Introduction */}
          <section id="introduction" className="thesis-section mb-16">
            <h2 className="thesis-h2">Introduction</h2>
            <p className="thesis-p">
              The design of neural network architectures has traditionally been a manual process,
              guided by researcher intuition and incremental refinement. While this approach has
              yielded remarkable results&mdash;from convolutional networks to transformers&mdash;the
              space of possible architectures remains vastly underexplored.
            </p>
            <p className="thesis-p">
              We propose using AI agents to systematically explore this design space, treating
              architecture design as a search problem where the objective function captures both
              model quality and computational efficiency.
            </p>
          </section>

          <section id="motivation" className="thesis-section mb-12">
            <h3 className="thesis-h3">Motivation</h3>
            <p className="thesis-p">
              The transformer architecture, introduced by Vaswani et al. (2017), has become the
              dominant paradigm in deep learning. Its core innovation&mdash;the self-attention
              mechanism&mdash;enables models to capture long-range dependencies in data. However,
              the quadratic complexity of self-attention with respect to sequence length presents
              significant computational challenges.
            </p>
            <div className="thesis-equation">
              <BlockMath math="\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V" />
            </div>
            <p className="thesis-p">
              The computational cost is dominated by the matrix product <InlineMath math="QK^\top \in \mathbb{R}^{n \times n}" />,
              which requires <InlineMath math="\mathcal{O}(n^2 d)" /> time and <InlineMath math="\mathcal{O}(n^2)" /> memory.
              This quadratic bottleneck has motivated a rich line of research into efficient
              alternatives: linear attention, sparse attention, low-rank approximations, and
              state-space models. Each offers different trade-offs between expressiveness and
              efficiency, but the space of possibilities extends far beyond what has been manually
              explored.
            </p>
          </section>

          <section id="core-hypothesis" className="thesis-section mb-12">
            <h3 className="thesis-h3">Core Hypothesis</h3>
            <div className="thesis-callout">
              <strong>Hypothesis:</strong> AI agents equipped with the ability to write, run, and
              evaluate code can discover novel architecture components that match or exceed
              human-designed alternatives, particularly in the efficiency&ndash;quality trade-off
              frontier.
            </div>
            <p className="thesis-p">
              More formally, let <InlineMath math="\mathcal{A}" /> denote the space of valid architectures and{' '}
              <InlineMath math="f: \mathcal{A} \to \mathbb{R}^k" /> a multi-objective evaluation function
              mapping architectures to quality and efficiency metrics. We seek to approximate the Pareto frontier:
            </p>
            <div className="thesis-equation">
              <BlockMath math="\mathcal{P}^* = \left\{ a \in \mathcal{A} \;\middle|\; \nexists\, a' \in \mathcal{A} : f(a') \succ f(a) \right\}" />
            </div>
            <p className="thesis-p">
              This hypothesis rests on several assumptions: (1) the architecture design space <InlineMath math="|\mathcal{A}|" /> is
              large enough that automated exploration can find regions missed by human designers,
              (2) proxy evaluation metrics are sufficiently correlated with downstream performance,
              and (3) AI agents can effectively navigate high-dimensional discrete search spaces
              through a combination of prior knowledge and systematic experimentation.
            </p>
          </section>

          {/* Architecture Search */}
          <section id="architecture-search" className="thesis-section mb-16">
            <h2 className="thesis-h2">Architecture Search with AI</h2>
            <p className="thesis-p">
              Our approach differs from traditional Neural Architecture Search (NAS) in a
              fundamental way: rather than searching over a predefined set of operations and
              connectivity patterns, we use language model agents to generate arbitrary PyTorch
              code. This dramatically expands the search space while leveraging the model's
              knowledge of existing architectures as an informed prior.
            </p>
            <p className="thesis-p">
              Traditional NAS methods parameterize the search space as a directed acyclic graph{' '}
              <InlineMath math="G = (V, E)" /> where nodes represent operations{' '}
              <InlineMath math="o_i \in \mathcal{O}" /> drawn from a fixed set. The search objective is typically:
            </p>
            <div className="thesis-equation">
              <BlockMath math="\alpha^* = \arg\min_{\alpha \in \mathcal{A}} \; \mathcal{L}_{\text{val}}\!\left(w^*(\alpha), \alpha\right) \quad \text{s.t.} \quad w^*(\alpha) = \arg\min_w \; \mathcal{L}_{\text{train}}(w, \alpha)" />
            </div>
            <p className="thesis-p">
              By contrast, our search space is the set of all programs <InlineMath math="p \in \mathcal{P}" /> that
              implement a valid <code>nn.Module</code>. The agent acts as a learned proposal
              distribution <InlineMath math="q_\theta(p \mid \mathcal{H})" /> conditioned on the history of
              prior experiments <InlineMath math="\mathcal{H} = \{(p_i, f(p_i))\}_{i=1}^t" />.
            </p>
          </section>

          <section id="search-space" className="thesis-section mb-12">
            <h3 className="thesis-h3">The Search Space</h3>
            <p className="thesis-p">
              The search space is implicitly defined by the set of all valid PyTorch modules that
              conform to a specified interface. An architecture component must accept inputs of a
              given shape and produce outputs of the expected shape, but the internal computation
              is unconstrained.
            </p>
            <div className="thesis-code">
              <pre>
                <code>{`class AttentionVariant(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # Architecture is unconstrained here
        ...

    def forward(self, x: Tensor) -> Tensor:
        # Must return tensor of same shape
        ...`}</code>
              </pre>
            </div>
            <p className="thesis-p">
              This formulation allows the agent to explore mechanisms that don't fit neatly into
              existing taxonomies&mdash;hybrid approaches that combine elements of attention,
              convolution, and recurrence in novel ways.
            </p>
          </section>

          <section id="evaluation-strategy" className="thesis-section mb-12">
            <h3 className="thesis-h3">Evaluation Strategy</h3>
            <p className="thesis-p">
              Each candidate architecture is evaluated on a suite of proxy tasks designed to
              measure both quality and efficiency. We define a composite score:
            </p>
            <div className="thesis-equation">
              <BlockMath math="S(a) = \underbrace{-\log \text{PPL}(a)}_{\text{quality}} + \lambda_1 \underbrace{\log \text{Throughput}(a)}_{\text{speed}} - \lambda_2 \underbrace{\log \text{Memory}(a)}_{\text{footprint}}" />
            </div>
            <p className="thesis-p">
              where <InlineMath math="\lambda_1, \lambda_2 > 0" /> control the efficiency&ndash;quality trade-off.
              The evaluation pipeline is fully automated:
            </p>
            <ol className="thesis-list">
              <li>
                <strong>Correctness:</strong> The module must compile and pass shape tests:
                verify <InlineMath math="f_a(x) \in \mathbb{R}^{B \times T \times d}" /> for
                all <InlineMath math="(B, T) \in \{1,4,8\} \times \{128, 512, 2048\}" />.
              </li>
              <li>
                <strong>Language modeling loss:</strong> Train on a small corpus (<InlineMath math="10^7" /> tokens) for{' '}
                <InlineMath math="N = 5000" /> steps and measure validation perplexity{' '}
                <InlineMath math="\text{PPL} = \exp\!\left(\frac{1}{|V|}\sum_{t} -\log p(x_t \mid x_{<t})\right)" />.
              </li>
              <li>
                <strong>Throughput:</strong> Measure tokens per second at sequence
                lengths <InlineMath math="T \in \{512, 1024, 2048, 4096\}" />.
              </li>
              <li>
                <strong>Memory footprint:</strong> Peak GPU memory <InlineMath math="M_{\text{peak}}" /> during
                training at each sequence length.
              </li>
            </ol>
            <p className="thesis-p">
              The agent uses these metrics to iteratively refine its proposals, building on
              successful experiments and learning from failures.
            </p>
          </section>

          {/* Attention Mechanisms */}
          <section id="attention-mechanisms" className="thesis-section mb-16">
            <h2 className="thesis-h2">Attention Mechanisms</h2>
            <p className="thesis-p">
              Understanding the landscape of existing attention mechanisms is essential for
              informed search. Here we review the key variants and their theoretical properties.
            </p>
          </section>

          <section id="softmax-attention" className="thesis-section mb-12">
            <h3 className="thesis-h3">Softmax Attention</h3>
            <p className="thesis-p">
              Standard softmax attention computes a weighted sum over values, where the weights
              are determined by the compatibility between queries and keys. Given input{' '}
              <InlineMath math="X \in \mathbb{R}^{n \times d}" />, we compute projections{' '}
              <InlineMath math="Q = XW_Q" />, <InlineMath math="K = XW_K" />, <InlineMath math="V = XW_V" /> and:
            </p>
            <div className="thesis-equation">
              <BlockMath math="\text{Attention}(Q, K, V)_i = \sum_{j=1}^{n} \frac{\exp\!\left(q_i^\top k_j / \sqrt{d_k}\right)}{\sum_{\ell=1}^{n} \exp\!\left(q_i^\top k_\ell / \sqrt{d_k}\right)} \, v_j" />
            </div>
            <p className="thesis-p">
              The softmax function ensures that attention weights are non-negative and sum to one,
              providing a probabilistic interpretation: <InlineMath math="\alpha_{ij} \geq 0" /> and{' '}
              <InlineMath math="\sum_j \alpha_{ij} = 1" />. However, this normalization couples all
              positions in the sequence, making the computation inherently quadratic.
            </p>
            <p className="thesis-p">
              For multi-head attention with <InlineMath math="h" /> heads, each head operates on a{' '}
              <InlineMath math="d_k = d/h" /> dimensional subspace. The outputs are concatenated and
              projected:
            </p>
            <div className="thesis-equation">
              <BlockMath math="\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_O \quad \text{where} \quad \text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)" />
            </div>
            <div className="thesis-callout">
              <strong>Complexity:</strong> <InlineMath math="\mathcal{O}(n^2 d)" /> time
              and <InlineMath math="\mathcal{O}(n^2)" /> memory, where <InlineMath math="n" /> is
              sequence length and <InlineMath math="d" /> is model dimension.
            </div>
          </section>

          <section id="linear-attention" className="thesis-section mb-12">
            <h3 className="thesis-h3">Linear Attention</h3>
            <p className="thesis-p">
              Linear attention replaces the softmax kernel with a decomposable feature
              map <InlineMath math="\phi: \mathbb{R}^d \to \mathbb{R}^D" />, enabling the computation to be
              rearranged via the associativity of matrix multiplication:
            </p>
            <div className="thesis-equation">
              <BlockMath math="\text{LinAttn}(Q, K, V)_i = \frac{\phi(q_i)^\top \sum_{j=1}^{n} \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_{j=1}^{n} \phi(k_j)} = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}" />
            </div>
            <p className="thesis-p">
              where <InlineMath math="S = \sum_j \phi(k_j) v_j^\top \in \mathbb{R}^{D \times d}" /> and{' '}
              <InlineMath math="z = \sum_j \phi(k_j) \in \mathbb{R}^D" /> can be computed once and shared
              across all query positions. This reduces the complexity from{' '}
              <InlineMath math="\mathcal{O}(n^2 d)" /> to <InlineMath math="\mathcal{O}(nDd)" />, which
              is linear in <InlineMath math="n" /> when <InlineMath math="D" /> is fixed.
            </p>
            <p className="thesis-p">
              For causal (autoregressive) modeling, we maintain a running state that can be updated recurrently:
            </p>
            <div className="thesis-equation">
              <BlockMath math="S_t = S_{t-1} + \phi(k_t) v_t^\top, \qquad z_t = z_{t-1} + \phi(k_t), \qquad o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}" />
            </div>
            <p className="thesis-p">
              Common choices for <InlineMath math="\phi" /> include the <InlineMath math="\text{elu}(x) + 1" /> activation
              (Katharopoulos et al., 2020), random Fourier features (Performer), and learned feature maps. Each introduces different
              trade-offs between approximation quality, training stability, and computational
              overhead.
            </p>
          </section>

          <section id="kernel-methods" className="thesis-section mb-12">
            <h3 className="thesis-h3">Kernel Methods</h3>
            <p className="thesis-p">
              The connection between attention and kernel methods provides a powerful theoretical
              framework. Softmax attention implicitly uses the kernel:
            </p>
            <div className="thesis-equation">
              <BlockMath math="k(x, y) = \exp\!\left(\frac{x^\top y}{\sqrt{d}}\right) = \exp\!\left(\frac{\|x\|^2}{2\sqrt{d}}\right) \exp\!\left(\frac{\|y\|^2}{2\sqrt{d}}\right) \exp\!\left(\frac{-\|x - y\|^2}{2\sqrt{d}}\right)" />
            </div>
            <p className="thesis-p">
              By Mercer's theorem, any positive-definite kernel admits a (possibly infinite-dimensional)
              feature map <InlineMath math="\phi" /> such that <InlineMath math="k(x,y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}" />.
              This opens the door to the rich theory of reproducing kernel
              Hilbert spaces (RKHS).
            </p>
            <p className="thesis-p">
              Of particular interest are polynomial kernels of degree <InlineMath math="p" />:
            </p>
            <div className="thesis-equation">
              <BlockMath math="k_p(x, y) = (1 + x^\top y)^p = \sum_{r=0}^{p} \binom{p}{r} (x^\top y)^r" />
            </div>
            <p className="thesis-p">
              These admit exact, finite-dimensional feature maps of
              dimension <InlineMath math="D = \binom{d+p}{p}" /> via the Veronese
              embedding. For <InlineMath math="p = 2" /> and <InlineMath math="d = 64" />,
              this gives <InlineMath math="D = 2145" />&mdash;tractable for linear attention with no
              approximation error.
            </p>
          </section>

          {/* Results */}
          <section id="experimental-results" className="thesis-section mb-16">
            <h2 className="thesis-h2">Experimental Results</h2>
            <p className="thesis-p">
              Our experiments span several hundred architecture proposals generated by AI agents,
              evaluated on the proxy task suite described above. Here we summarize the key
              findings.
            </p>
          </section>

          <section id="scaling-laws" className="thesis-section mb-12">
            <h3 className="thesis-h3">Scaling Laws</h3>
            <p className="thesis-p">
              Following Kaplan et al. (2020), we model validation loss as a power law in compute:
            </p>
            <div className="thesis-equation">
              <BlockMath math="L(C) = L_\infty + \left(\frac{C_0}{C}\right)^\beta" />
            </div>
            <p className="thesis-p">
              where <InlineMath math="C" /> is training compute in FLOPs, <InlineMath math="L_\infty" /> is the
              irreducible loss, and <InlineMath math="\beta" /> is the scaling exponent. We fit this for each
              architecture variant and find that the best agent-discovered architectures
              achieve <InlineMath math="\beta \approx 0.076" />, comparable to the
              transformer baseline (<InlineMath math="\beta_{\text{transformer}} \approx 0.079" />),
              while offering 2&ndash;4x throughput improvements at sequence length 4096.
            </p>
            <div className="thesis-figure">
              <div className="thesis-figure-placeholder">
                [Figure: Scaling curves for top-5 discovered architectures vs. baseline transformer.
                X-axis: training compute <InlineMath math="C" /> (log scale). Y-axis: validation loss <InlineMath math="L(C)" />.]
              </div>
            </div>
            <p className="thesis-p">
              Interestingly, the scaling exponents <InlineMath math="\beta" /> differ between attention variants. Linear
              attention variants tend to show larger <InlineMath math="\beta" /> (steeper initial improvement)
              but higher <InlineMath math="L_\infty" /> (earlier saturation),
              while hybrid approaches maintain more consistent scaling with{' '}
              <InlineMath math="L_\infty" /> values within 3% of the transformer baseline.
            </p>
          </section>

          <section id="efficiency-benchmarks" className="thesis-section mb-12">
            <h3 className="thesis-h3">Efficiency Benchmarks</h3>
            <p className="thesis-p">
              The efficiency&ndash;quality Pareto frontier reveals several clusters of
              architectures:
            </p>
            <div className="thesis-table-wrapper">
              <table className="thesis-table">
                <thead>
                  <tr>
                    <th>Architecture</th>
                    <th>Perplexity</th>
                    <th>Throughput (tok/s)</th>
                    <th>Memory (GB)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Transformer (baseline)</td>
                    <td>24.3</td>
                    <td>12,400</td>
                    <td>8.2</td>
                  </tr>
                  <tr>
                    <td>Linear Attention (ELU)</td>
                    <td>28.1</td>
                    <td>31,200</td>
                    <td>3.1</td>
                  </tr>
                  <tr>
                    <td>Agent Discovery A</td>
                    <td>25.1</td>
                    <td>24,800</td>
                    <td>4.7</td>
                  </tr>
                  <tr>
                    <td>Agent Discovery B</td>
                    <td>24.8</td>
                    <td>19,600</td>
                    <td>5.3</td>
                  </tr>
                  <tr>
                    <td>Agent Discovery C</td>
                    <td>26.4</td>
                    <td>28,100</td>
                    <td>3.4</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="thesis-p">
              Agent Discovery B is particularly noteworthy: it achieves near-baseline perplexity
              with 58% higher throughput, using a novel combination of local windowed attention
              with a learned gating mechanism that selectively engages a lightweight global
              attention path.
            </p>
          </section>

          {/* Future */}
          <section id="future-directions" className="thesis-section mb-16">
            <h2 className="thesis-h2">Future Directions</h2>
            <p className="thesis-p">
              Several promising directions emerge from this work:
            </p>
            <ol className="thesis-list">
              <li>
                <strong>Compositional search:</strong> Rather than generating monolithic attention
                modules, decompose the search into composable primitives (projection, aggregation,
                normalization, gating) and search over compositions.
              </li>
              <li>
                <strong>Transfer across scales:</strong> Develop better proxy tasks that predict
                large-scale performance from small-scale experiments, potentially using learned
                scaling law models.
              </li>
              <li>
                <strong>Multi-objective optimization:</strong> Explicitly optimize the
                Pareto frontier using scalarization <InlineMath math="\min_a \max_\lambda \sum_i \lambda_i f_i(a)" /> across
                quality, latency, memory, and hardware utilization.
              </li>
              <li>
                <strong>Theoretical analysis:</strong> Use the discovered architectures to develop
                new theoretical understanding of why certain computational patterns are effective
                for sequence modeling.
              </li>
            </ol>
            <p className="thesis-p">
              The broader vision is a tighter loop between theory, implementation, and empirical
              evaluation&mdash;where AI agents serve as tireless research assistants, systematically
              testing hypotheses and surfacing patterns that guide human understanding.
            </p>
          </section>

          <footer className="mt-20 pt-8 border-t" style={{ borderColor: 'var(--paper-deep)' }}>
            <p className="text-sm" style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-muted)' }}>
              These notes are a living document, updated as new experiments complete and new
              insights emerge. For the latest experimental results, see the{' '}
              <a href="/" className="underline" style={{ color: 'var(--accent)' }}>
                platform dashboard
              </a>
              .
            </p>
          </footer>
        </article>
      </div>
    </div>
  )
}
