import { Link } from 'react-router-dom'

function Endpoint({
  method,
  path,
  description,
  children,
}: {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
  path: string
  description: string
  children?: React.ReactNode
}) {
  const methodColors: Record<string, string> = {
    GET: '#22c55e',
    POST: '#3b82f6',
    PUT: '#f59e0b',
    PATCH: '#f59e0b',
    DELETE: '#ef4444',
  }
  return (
    <div
      className="rounded-lg border p-6"
      style={{ borderColor: 'var(--paper-deep)', backgroundColor: 'var(--paper)' }}
    >
      <div className="flex items-center gap-3 mb-3">
        <span
          className="text-xs font-bold px-2 py-1 rounded"
          style={{
            fontFamily: 'var(--font-mono, monospace)',
            color: '#fff',
            backgroundColor: methodColors[method] || 'var(--ink-muted)',
          }}
        >
          {method}
        </span>
        <code
          className="text-sm"
          style={{ fontFamily: 'var(--font-mono, monospace)', color: 'var(--ink)' }}
        >
          {path}
        </code>
      </div>
      <p
        className="leading-relaxed mb-3 text-sm"
        style={{ color: 'var(--ink-muted)' }}
      >
        {description}
      </p>
      {children}
    </div>
  )
}

function CodeBlock({ children }: { children: string }) {
  return (
    <pre
      className="text-xs leading-relaxed overflow-x-auto rounded p-4"
      style={{
        fontFamily: 'var(--font-mono, monospace)',
        backgroundColor: 'var(--paper-deep)',
        color: 'var(--ink)',
      }}
    >
      {children}
    </pre>
  )
}

export default function MarketMakingPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--paper)' }}>
      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-8">
          <Link
            to="/engineering"
            className="text-sm hover:underline"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink-muted)' }}
          >
            &larr; Engineering
          </Link>
        </div>

        <header className="mb-12">
          <h1
            className="text-3xl font-bold mb-4 leading-tight"
            style={{ fontFamily: 'var(--font-display)', color: 'var(--ink)' }}
          >
            Market Making API
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            The API for creating conjectures, trading shares, submitting
            evidence, and tracking portfolios.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>

          {/* --- Overview --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Overview
          </h2>

          <p className="leading-relaxed">
            The Market Making API is a RESTful JSON API. All requests
            require authentication via bearer token. All timestamps are
            ISO 8601 UTC. All monetary values are in the market&rsquo;s
            internal unit of account.
          </p>

          <CodeBlock>{`Base URL: https://api.silon.io/v1

Authentication:
  Authorization: Bearer sk_live_...`}</CodeBlock>

          {/* --- Entity IDs --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Entity IDs
          </h2>

          <p className="leading-relaxed">
            Every entity in the system has a unique identifier with a
            type-specific prefix, making it immediately clear what kind of
            object you are looking at in logs, URLs, and payloads.
          </p>

          <div
            className="rounded-lg border overflow-hidden"
            style={{ borderColor: 'var(--paper-deep)' }}
          >
            <table className="w-full text-sm">
              <thead>
                <tr style={{ backgroundColor: 'var(--paper-deep)' }}>
                  <th className="text-left px-4 py-3 font-semibold" style={{ fontFamily: 'var(--font-display)' }}>Entity</th>
                  <th className="text-left px-4 py-3 font-semibold" style={{ fontFamily: 'var(--font-display)' }}>Prefix</th>
                  <th className="text-left px-4 py-3 font-semibold" style={{ fontFamily: 'var(--font-display)' }}>Example</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderTop: '1px solid var(--paper-deep)' }}>
                  <td className="px-4 py-3">Conjecture</td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>conj_</code></td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>conj_8f3a2b1c4d5e</code></td>
                </tr>
                <tr style={{ borderTop: '1px solid var(--paper-deep)' }}>
                  <td className="px-4 py-3">Participant</td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>part_</code></td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>part_a1b2c3d4e5f6</code></td>
                </tr>
                <tr style={{ borderTop: '1px solid var(--paper-deep)' }}>
                  <td className="px-4 py-3">Trade</td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>trd_</code></td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>trd_7e6f5a4b3c2d</code></td>
                </tr>
                <tr style={{ borderTop: '1px solid var(--paper-deep)' }}>
                  <td className="px-4 py-3">Bundle</td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>bndl_</code></td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>bndl_2d3e4f5a6b7c</code></td>
                </tr>
                <tr style={{ borderTop: '1px solid var(--paper-deep)' }}>
                  <td className="px-4 py-3">Evidence</td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>evd_</code></td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>evd_9c8b7a6f5e4d</code></td>
                </tr>
                <tr style={{ borderTop: '1px solid var(--paper-deep)' }}>
                  <td className="px-4 py-3">Portfolio</td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>ptf_</code></td>
                  <td className="px-4 py-3"><code style={{ fontFamily: 'var(--font-mono, monospace)' }}>ptf_1a2b3c4d5e6f</code></td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="leading-relaxed mt-4">
            IDs are 12-character lowercase hexadecimal strings with a
            type prefix. They are globally unique and immutable.
          </p>

          {/* --- Conjectures --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Conjectures
          </h2>

          <div className="space-y-4">
            <Endpoint
              method="POST"
              path="/v1/conjectures"
              description="Create a new conjecture. It enters the market at credence 0.50 with maximum entropy."
            >
              <CodeBlock>{`{
  "statement": "CRISPR base editing achieves >90% efficiency in primary human T cells",
  "tags": ["biology", "gene-editing", "crispr"],
  "resolution_criteria": "Published, peer-reviewed benchmark showing >=90% editing efficiency in primary human T cells using any base editor.",
  "visibility": "public"
}`}</CodeBlock>
              <p className="text-xs mt-3" style={{ color: 'var(--ink-muted)' }}>
                Returns the created conjecture with its <code style={{ fontFamily: 'var(--font-mono, monospace)' }}>conj_</code> ID, initial credence (0.50), and current entropy (1.0 bit).
              </p>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/conjectures/:id"
              description="Retrieve a conjecture by ID. Returns the current credence, entropy, trade volume, and dependency graph edges."
            >
              <CodeBlock>{`// Response
{
  "id": "conj_8f3a2b1c4d5e",
  "statement": "CRISPR base editing achieves >90% efficiency in primary human T cells",
  "credence": 0.55,
  "entropy": 0.99,
  "created_at": "2026-01-15T10:30:00Z",
  "trade_count": 142,
  "dependencies": ["conj_a1b2c3d4e5f6", "conj_7e6f5a4b3c2d"],
  "dependents": ["conj_2d3e4f5a6b7c"],
  "visibility": "public"
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/conjectures"
              description="List conjectures. Supports filtering by tag, credence range, entropy range, and sorting by entropy (highest first for research targets)."
            >
              <CodeBlock>{`GET /v1/conjectures?tag=biology&min_entropy=0.8&sort=entropy_desc&limit=20`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/conjectures/:id/history"
              description="Retrieve the credence and entropy time series for a conjecture. Returns data points at each trade or evidence event."
            >
              <CodeBlock>{`// Response
{
  "conjecture_id": "conj_8f3a2b1c4d5e",
  "series": [
    { "t": "2026-01-15T10:30:00Z", "credence": 0.50, "entropy": 1.0 },
    { "t": "2026-01-15T11:02:00Z", "credence": 0.55, "entropy": 0.99 },
    { "t": "2026-02-03T09:15:00Z", "credence": 0.62, "entropy": 0.96 },
    ...
  ]
}`}</CodeBlock>
            </Endpoint>
          </div>

          {/* --- Trades --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Trades
          </h2>

          <div className="space-y-4">
            <Endpoint
              method="POST"
              path="/v1/trades"
              description="Execute a trade. Specify direction (YES/NO), quantity, the conjecture, and optionally a bundle and evidence. The cost is direction-weighted: P(A) * H(P) for YES, (1-P(A)) * H(P) for NO."
            >
              <CodeBlock>{`{
  "conjecture_id": "conj_8f3a2b1c4d5e",
  "direction": "YES",
  "quantity": 10,
  "bundle_id": "bndl_2d3e4f5a6b7c",
  "evidence_id": "evd_9c8b7a6f5e4d",
  "trade_visibility": "public",
  "evidence_visibility": "public"
}`}</CodeBlock>
              <p className="text-xs mt-3" style={{ color: 'var(--ink-muted)' }}>
                Both <code style={{ fontFamily: 'var(--font-mono, monospace)' }}>bundle_id</code> and <code style={{ fontFamily: 'var(--font-mono, monospace)' }}>evidence_id</code> are optional. Visibility defaults to <code style={{ fontFamily: 'var(--font-mono, monospace)' }}>private</code> if omitted.
              </p>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/trades/:id"
              description="Retrieve a trade by ID. Returns the trade details, cost paid, entropy at entry, and current directional reward."
            >
              <CodeBlock>{`// Response
{
  "id": "trd_7e6f5a4b3c2d",
  "conjecture_id": "conj_8f3a2b1c4d5e",
  "participant_id": "part_a1b2c3d4e5f6",
  "direction": "YES",
  "quantity": 10,
  "entry_credence": 0.30,
  "entry_entropy": 0.88,
  "cost": 8.80,
  "current_reward": 2.20,
  "bundle_id": "bndl_2d3e4f5a6b7c",
  "evidence_id": "evd_9c8b7a6f5e4d",
  "trade_visibility": "public",
  "evidence_visibility": "public",
  "created_at": "2026-03-01T14:22:00Z"
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/conjectures/:id/trades"
              description="List all public trades on a conjecture, ordered by time. Useful for viewing the trade history and evidence trail."
            />
          </div>

          {/* --- Bundles --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Bundles
          </h2>

          <div className="space-y-4">
            <Endpoint
              method="POST"
              path="/v1/bundles"
              description="Create a bundle — a set of conjectures you believe are logically connected. When you trade with a bundle, you buy all included conjectures and may receive a Bayesian discount on each."
            >
              <CodeBlock>{`{
  "conjecture_ids": [
    "conj_8f3a2b1c4d5e",
    "conj_a1b2c3d4e5f6",
    "conj_7e6f5a4b3c2d"
  ]
}`}</CodeBlock>
              <p className="text-xs mt-3" style={{ color: 'var(--ink-muted)' }}>
                Returns the bundle with its <code style={{ fontFamily: 'var(--font-mono, monospace)' }}>bndl_</code> ID and the estimated conditional entropy discount for each conjecture given the others.
              </p>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/bundles/:id"
              description="Retrieve a bundle by ID. Returns the included conjectures and the current discount estimates."
            >
              <CodeBlock>{`// Response
{
  "id": "bndl_2d3e4f5a6b7c",
  "conjecture_ids": [
    "conj_8f3a2b1c4d5e",
    "conj_a1b2c3d4e5f6",
    "conj_7e6f5a4b3c2d"
  ],
  "discounts": {
    "conj_8f3a2b1c4d5e": { "marginal_entropy": 0.99, "conditional_entropy": 0.72, "discount": 0.27 },
    "conj_a1b2c3d4e5f6": { "marginal_entropy": 0.88, "conditional_entropy": 0.65, "discount": 0.23 },
    "conj_7e6f5a4b3c2d": { "marginal_entropy": 0.95, "conditional_entropy": 0.80, "discount": 0.15 }
  },
  "created_at": "2026-02-20T08:45:00Z"
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/participants/:id/bundles"
              description="List all bundles owned by a participant."
            />
          </div>

          {/* --- Evidence --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Evidence
          </h2>

          <div className="space-y-4">
            <Endpoint
              method="POST"
              path="/v1/evidence"
              description="Submit evidence for one or more conjectures. Evidence can be a paper, dataset, experimental result, or argument. Attach it to trades to explain why you are trading."
            >
              <CodeBlock>{`{
  "title": "High-fidelity Cas9 variant achieves 94% base editing in T cells",
  "type": "paper",
  "url": "https://doi.org/10.1234/example.2026.001",
  "conjecture_ids": ["conj_8f3a2b1c4d5e"],
  "direction": "supports",
  "summary": "New Cas9 variant with engineered PAM flexibility shows 94% editing efficiency in primary human T cells across three independent replicates.",
  "visibility": "public"
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/evidence/:id"
              description="Retrieve evidence by ID. Returns the metadata, linked conjectures, and the credence impact (how much the evidence moved the credence when submitted)."
            >
              <CodeBlock>{`// Response
{
  "id": "evd_9c8b7a6f5e4d",
  "title": "High-fidelity Cas9 variant achieves 94% base editing in T cells",
  "type": "paper",
  "url": "https://doi.org/10.1234/example.2026.001",
  "conjecture_ids": ["conj_8f3a2b1c4d5e"],
  "direction": "supports",
  "credence_impact": {
    "conj_8f3a2b1c4d5e": { "before": 0.30, "after": 0.55, "delta": 0.25 }
  },
  "submitted_by": "part_a1b2c3d4e5f6",
  "visibility": "public",
  "created_at": "2026-03-01T14:20:00Z"
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/conjectures/:id/evidence"
              description="List all public evidence submitted for a conjecture, ordered by time."
            />
          </div>

          {/* --- Portfolios --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Portfolios
          </h2>

          <div className="space-y-4">
            <Endpoint
              method="GET"
              path="/v1/participants/:id/portfolio"
              description="Retrieve a participant's portfolio. Returns all positions with entry credence, entry entropy, current credence, and directional reward per position."
            >
              <CodeBlock>{`// Response
{
  "participant_id": "part_a1b2c3d4e5f6",
  "portfolio_value": 47.32,
  "positions": [
    {
      "conjecture_id": "conj_8f3a2b1c4d5e",
      "direction": "YES",
      "quantity": 50,
      "entry_credence": 0.30,
      "entry_entropy": 0.88,
      "current_credence": 0.55,
      "current_entropy": 0.99,
      "directional_reward": 11.0
    },
    {
      "conjecture_id": "conj_a1b2c3d4e5f6",
      "direction": "YES",
      "quantity": 30,
      "entry_credence": 0.60,
      "entry_entropy": 0.97,
      "current_credence": 0.65,
      "current_entropy": 0.93,
      "directional_reward": 1.46
    }
  ]
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/participants/:id/portfolio/history"
              description="Portfolio value time series. Returns the total portfolio value at each trade or evidence event, for charting the participant's track record."
            >
              <CodeBlock>{`// Response
{
  "participant_id": "part_a1b2c3d4e5f6",
  "series": [
    { "t": "2026-01-15T10:30:00Z", "value": 0.0 },
    { "t": "2026-01-15T11:02:00Z", "value": 2.20 },
    { "t": "2026-03-01T14:22:00Z", "value": 12.46 },
    { "t": "2026-03-15T09:00:00Z", "value": 47.32 },
    ...
  ]
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/participants/:id/impact"
              description="Retrieve a participant's total conjecture impact score, with breakdown by conjecture and direction (upstream vs. downstream)."
            />
          </div>

          {/* --- Dependency Graph --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Dependency Graph
          </h2>

          <div className="space-y-4">
            <Endpoint
              method="GET"
              path="/v1/graph/neighbors/:conjecture_id"
              description="Retrieve the immediate dependencies and dependents of a conjecture, with edge confidence scores derived from bundle co-occurrence and evidence propagation."
            >
              <CodeBlock>{`// Response
{
  "conjecture_id": "conj_8f3a2b1c4d5e",
  "dependencies": [
    { "id": "conj_a1b2c3d4e5f6", "confidence": 0.87, "bundle_support": 142, "propagation_support": 0.72 }
  ],
  "dependents": [
    { "id": "conj_2d3e4f5a6b7c", "confidence": 0.63, "bundle_support": 58, "propagation_support": 0.41 }
  ]
}`}</CodeBlock>
            </Endpoint>

            <Endpoint
              method="GET"
              path="/v1/graph/path/:from/:to"
              description="Find the shortest dependency path between two conjectures, if one exists. Returns the path with edge confidences."
            />

            <Endpoint
              method="GET"
              path="/v1/graph/bottlenecks"
              description="List high-entropy conjectures with many downstream dependents — the upstream bottlenecks whose resolution would have the highest information cascade. Useful for identifying high-value research targets."
            >
              <CodeBlock>{`GET /v1/graph/bottlenecks?min_dependents=5&min_entropy=0.7&limit=10`}</CodeBlock>
            </Endpoint>
          </div>

          {/* --- Errors --- */}
          <h2
            className="text-2xl font-bold mt-10"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Errors
          </h2>

          <p className="leading-relaxed">
            The API uses standard HTTP status codes. Error responses include
            a machine-readable <code style={{ fontFamily: 'var(--font-mono, monospace)' }}>code</code> and
            a human-readable <code style={{ fontFamily: 'var(--font-mono, monospace)' }}>message</code>.
          </p>

          <CodeBlock>{`// 400 Bad Request
{
  "error": {
    "code": "invalid_direction",
    "message": "Direction must be 'YES' or 'NO'."
  }
}

// 402 Payment Required
{
  "error": {
    "code": "insufficient_balance",
    "message": "Trade cost (8.80) exceeds available balance (3.12)."
  }
}

// 404 Not Found
{
  "error": {
    "code": "conjecture_not_found",
    "message": "No conjecture with ID conj_000000000000."
  }
}`}</CodeBlock>

        </div>
      </div>
    </div>
  )
}
