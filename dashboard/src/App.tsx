import { Routes, Route } from 'react-router-dom'
import { MDXProvider } from '@mdx-js/react'
import Layout from './components/Layout'
import LandingPage from './pages/LandingPage'
import MADDashboard from './pages/MADDashboard'
import ThesisPage from './pages/ThesisPage'
import PlatformPage from './pages/PlatformPage'
import HelloWorldPage from './pages/platform/HelloWorldPage'
import CreatingConjecturesPage from './pages/platform/CreatingConjecturesPage'
import IntegrationsPage from './pages/platform/IntegrationsPage'
import OpenProblemsPage from './pages/platform/OpenProblemsPage'
import ScoringMetricsPage from './pages/platform/ScoringMetricsPage'
import ExampleConjecturesPage from './pages/platform/ExampleConjecturesPage'
import ShortingPage from './pages/platform/ShortingPage'
import MarketIncentivesPage from './pages/platform/MarketIncentivesPage'
import PriceDeterminationPage from './pages/platform/PriceDeterminationPage'
import { mdxComponents } from './components/MDXComponents'

function App() {
  return (
    <MDXProvider components={mdxComponents}>
      <Routes>
        <Route path="/" element={<Layout><LandingPage /></Layout>} />
        <Route path="/thesis" element={<Layout><ThesisPage /></Layout>} />
        <Route path="/platform" element={<Layout><PlatformPage /></Layout>} />
        <Route path="/platform/hello-world" element={<Layout><HelloWorldPage /></Layout>} />
        <Route path="/platform/scoring-metrics" element={<Layout><ScoringMetricsPage /></Layout>} />
        <Route path="/platform/creating-conjectures" element={<Layout><CreatingConjecturesPage /></Layout>} />
        <Route path="/platform/integrations" element={<Layout><IntegrationsPage /></Layout>} />
        <Route path="/platform/example-conjectures" element={<Layout><ExampleConjecturesPage /></Layout>} />
        <Route path="/platform/market-incentives" element={<Layout><MarketIncentivesPage /></Layout>} />
        <Route path="/platform/shorting" element={<Layout><ShortingPage /></Layout>} />
        <Route path="/platform/price-determination" element={<Layout><PriceDeterminationPage /></Layout>} />
        <Route path="/platform/open-problems" element={<Layout><OpenProblemsPage /></Layout>} />
        <Route path="/agent/*" element={<Layout><div className="max-w-3xl mx-auto px-4 py-8"><MADDashboard /></div></Layout>} />
      </Routes>
    </MDXProvider>
  )
}

export default App
