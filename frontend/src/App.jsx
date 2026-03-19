import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './Navbar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Chat from './pages/Chat'
import Goals from './pages/Goals'
import Onboarding from './pages/Onboarding'

const AppLayout = ({ children }) => (
  <>
    <Navbar />
    {children}
  </>
)

const ProtectedApp = ({ children }) => {
  const done = localStorage.getItem('onboarding_complete')
  if (!done) return <Navigate to="/onboarding" replace />
  return children
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/app" element={<Navigate to="/app/upload" replace />} />
        <Route path="/app/upload" element={
          <ProtectedApp><AppLayout><Upload /></AppLayout></ProtectedApp>
        }/>
        <Route path="/app/dashboard" element={
          <ProtectedApp><AppLayout><Dashboard /></AppLayout></ProtectedApp>
        }/>
        <Route path="/app/chat" element={
          <ProtectedApp><AppLayout><Chat /></AppLayout></ProtectedApp>
        }/>
        <Route path="/app/goals" element={
          <ProtectedApp><AppLayout><Goals /></AppLayout></ProtectedApp>
        }/>
      </Routes>
    </BrowserRouter>
  )
}
