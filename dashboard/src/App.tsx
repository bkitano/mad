import { Routes, Route } from 'react-router-dom'
import { MDXProvider } from '@mdx-js/react'
import Layout from './components/Layout'
import MADDashboard from './pages/MADDashboard'
import { mdxComponents } from './components/MDXComponents'

function App() {
  return (
    <MDXProvider components={mdxComponents}>
      <Routes>
        {/* Standard pages with layout */}
        <Route path="/mad/*" element={<Layout><MADDashboard /></Layout>} />
      </Routes>
    </MDXProvider>
  )
}

export default App
