import { Routes, Route } from 'react-router-dom'
import { MDXProvider } from '@mdx-js/react'
import Layout from './components/Layout'
import MADDashboard from './pages/MADDashboard'
import ThesisPage from './pages/ThesisPage'
import { mdxComponents } from './components/MDXComponents'

function App() {
  return (
    <MDXProvider components={mdxComponents}>
      <Routes>
        <Route path="/thesis" element={<Layout><ThesisPage /></Layout>} />
        <Route path="/*" element={<Layout><div className="max-w-3xl mx-auto px-4 py-8"><MADDashboard /></div></Layout>} />
      </Routes>
    </MDXProvider>
  )
}

export default App
