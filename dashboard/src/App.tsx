import { Routes, Route } from 'react-router-dom'
import { MDXProvider } from '@mdx-js/react'
import Layout from './components/Layout'
import LandingPage from './pages/LandingPage'
import MADDashboard from './pages/MADDashboard'
import ThesisPage from './pages/ThesisPage'
import PlatformPage from './pages/PlatformPage'
import HelloWorldPage from './pages/platform/HelloWorldPage'
import { mdxComponents } from './components/MDXComponents'

function App() {
  return (
    <MDXProvider components={mdxComponents}>
      <Routes>
        <Route path="/" element={<Layout><LandingPage /></Layout>} />
        <Route path="/thesis" element={<Layout><ThesisPage /></Layout>} />
        <Route path="/platform" element={<Layout><PlatformPage /></Layout>} />
        <Route path="/platform/hello-world" element={<Layout><HelloWorldPage /></Layout>} />
        <Route path="/agent/*" element={<Layout><div className="max-w-3xl mx-auto px-4 py-8"><MADDashboard /></div></Layout>} />
      </Routes>
    </MDXProvider>
  )
}

export default App
