import { Link } from 'react-router-dom'

export default function IntegrationsPage() {
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
            Integrations
          </h1>
          <p
            className="text-lg leading-relaxed"
            style={{ fontFamily: 'var(--font-body)', color: 'var(--ink-muted)' }}
          >
            Connecting the conjecture market to existing scientific infrastructure.
          </p>
        </header>

        <div className="space-y-6" style={{ fontFamily: 'var(--font-body)', color: 'var(--ink)' }}>
          <p className="leading-relaxed">
            The conjecture market does not exist in a vacuum. The vast majority of
            scientific work is already published, indexed, and discussed through existing
            venues&mdash;arXiv, peer-reviewed journals, Google Scholar, Semantic Scholar,
            and others. For the market to be useful, it needs to meet science where it
            already happens.
          </p>

          <p className="leading-relaxed">
            We are exploring integrations along several axes:
          </p>

          <ul className="space-y-4">
            <li className="leading-relaxed">
              <strong>arXiv.</strong> New preprints are a primary source of evidence that
              moves conjecture prices. We plan to ingest arXiv submissions and surface
              conjectures that a given paper is relevant to, making it easy for participants
              to connect new work to existing market positions.
            </li>
            <li className="leading-relaxed">
              <strong>Google Scholar and Semantic Scholar.</strong> Citation graphs encode
              the dependency structure of science&mdash;which results build on which. These
              graphs can seed the conjecture dependency network, and citation events can
              serve as weak signals for price updates.
            </li>
            <li className="leading-relaxed">
              <strong>Peer-reviewed publications.</strong> Acceptance at a venue is a signal
              (noisy, but real) that a result has passed a verification threshold. We are
              thinking about how publication events should interact with conjecture
              prices&mdash;not as an oracle, but as one more piece of evidence the market
              can incorporate.
            </li>
          </ul>

          <p className="leading-relaxed">
            The goal is not to replace these systems but to layer a market mechanism on
            top of them&mdash;one that aggregates the signals they already produce into
            a faster, more legible measure of scientific credence.
          </p>
        </div>
      </div>
    </div>
  )
}
