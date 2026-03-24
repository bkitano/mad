import { useState, useEffect } from 'react'

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
              <code>Attention(Q, K, V) = softmax(QK&sup1; / &radic;d) &middot; V</code>
            </div>
            <p className="thesis-p">
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
              This hypothesis rests on several assumptions: (1) the architecture design space is
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
              measure both quality and efficiency. The evaluation pipeline is fully automated:
            </p>
            <ol className="thesis-list">
              <li>
                <strong>Correctness:</strong> The module must compile and pass shape tests across
                a range of input dimensions and batch sizes.
              </li>
              <li>
                <strong>Language modeling loss:</strong> Train on a small corpus (10M tokens) for a
                fixed number of steps and measure validation perplexity.
              </li>
              <li>
                <strong>Throughput:</strong> Measure tokens processed per second at various
                sequence lengths (512, 1024, 2048, 4096).
              </li>
              <li>
                <strong>Memory footprint:</strong> Peak GPU memory usage during training at each
                sequence length.
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
              are determined by the compatibility between queries and keys:
            </p>
            <div className="thesis-equation">
              <code>
                A(Q, K, V)&#8582; = &sum;&#8645; softmax(Q&#8582;K&sup1; / &radic;d)&#8582;&#8645;
                &middot; V&#8645;
              </code>
            </div>
            <p className="thesis-p">
              The softmax function ensures that attention weights are non-negative and sum to one,
              providing a probabilistic interpretation. However, this normalization couples all
              positions in the sequence, making the computation inherently sequential in the
              attention dimension and quadratic in sequence length.
            </p>
            <div className="thesis-callout">
              <strong>Complexity:</strong> O(n&sup2;d) time and O(n&sup2;) memory, where n is
              sequence length and d is head dimension.
            </div>
          </section>

          <section id="linear-attention" className="thesis-section mb-12">
            <h3 className="thesis-h3">Linear Attention</h3>
            <p className="thesis-p">
              Linear attention replaces the softmax with a decomposable kernel function
              &phi;, enabling the computation to be rearranged using the associativity of matrix
              multiplication:
            </p>
            <div className="thesis-equation">
              <code>
                A(Q, K, V)&#8582; = &phi;(Q&#8582;) &middot; (&sum;&#8645; &phi;(K&#8645;)&sup1;
                V&#8645;) / (&phi;(Q&#8582;) &middot; &sum;&#8645; &phi;(K&#8645;))
              </code>
            </div>
            <p className="thesis-p">
              The key insight is that the term &sum;&#8645; &phi;(K&#8645;)&sup1; V&#8645; can be
              computed once and shared across all query positions, reducing the complexity from
              quadratic to linear. The choice of feature map &phi; determines the approximation
              quality and computational characteristics.
            </p>
            <p className="thesis-p">
              Common choices include the ELU+1 activation (Katharopoulos et al., 2020), random
              Fourier features (Performer), and learned feature maps. Each introduces different
              trade-offs between approximation quality, training stability, and computational
              overhead.
            </p>
          </section>

          <section id="kernel-methods" className="thesis-section mb-12">
            <h3 className="thesis-h3">Kernel Methods</h3>
            <p className="thesis-p">
              The connection between attention and kernel methods provides a powerful theoretical
              framework. The softmax attention kernel can be expressed as:
            </p>
            <div className="thesis-equation">
              <code>k(x, y) = exp(x &middot; y / &radic;d)</code>
            </div>
            <p className="thesis-p">
              This perspective suggests that attention mechanisms can be understood as
              kernel-weighted averaging, opening the door to the rich theory of reproducing kernel
              Hilbert spaces (RKHS). Different kernels induce different notions of similarity and
              different computational trade-offs.
            </p>
            <p className="thesis-p">
              Of particular interest are polynomial kernels and Maclaurin expansions, which provide
              finite-dimensional feature maps and thus exact (rather than approximate) linear
              attention with controllable complexity.
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
              We observe consistent scaling behavior across architecture variants. When controlling
              for parameter count, the best agent-discovered architectures achieve validation
              perplexity within 5% of standard transformers at sequence lengths up to 2048, while
              offering 2&ndash;4x throughput improvements at sequence length 4096.
            </p>
            <div className="thesis-figure">
              <div className="thesis-figure-placeholder">
                [Figure: Scaling curves for top-5 discovered architectures vs. baseline transformer.
                X-axis: training tokens (log scale). Y-axis: validation perplexity.]
              </div>
            </div>
            <p className="thesis-p">
              Interestingly, the scaling exponents differ between attention variants. Linear
              attention variants tend to show steeper initial improvement but earlier saturation,
              while hybrid approaches maintain more consistent scaling across the range of compute
              budgets tested.
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
                Pareto frontier across quality, latency, memory, and hardware utilization,
                rather than treating efficiency as a constraint.
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
