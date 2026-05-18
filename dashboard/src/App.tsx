import { Routes, Route } from 'react-router-dom'
import { MDXProvider } from '@mdx-js/react'
import Layout from './components/Layout'
import ProtectedRoute from './components/ProtectedRoute'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import AuthCallbackPage from './pages/AuthCallbackPage'
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
import EngineeringPage from './pages/EngineeringPage'
import BacktestingPage from './pages/engineering/BacktestingPage'
import MarketMakingPage from './pages/engineering/MarketMakingPage'
import BayesianNetworksPage from './pages/platform/BayesianNetworksPage'
import { mdxComponents } from './components/MDXComponents'

function App() {
  return (
    <MDXProvider components={mdxComponents}>
      <Routes>
        <Route path="/" element={<Layout><LandingPage /></Layout>} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/auth/callback" element={<AuthCallbackPage />} />
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
        <Route path="/platform/bayesian-networks" element={<Layout><BayesianNetworksPage /></Layout>} />
        <Route path="/platform/open-problems" element={<Layout><OpenProblemsPage /></Layout>} />
        <Route path="/engineering" element={<Layout><EngineeringPage /></Layout>} />
        <Route path="/engineering/backtesting" element={<Layout><BacktestingPage /></Layout>} />
        <Route path="/engineering/market-making" element={<Layout><MarketMakingPage /></Layout>} />
        <Route path="/agent/*" element={<ProtectedRoute><MADDashboard /></ProtectedRoute>} />
      </Routes>
    </MDXProvider>
  )
}

export default App
