import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './Navbar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Chat from './pages/Chat'
import Goals from './pages/Goals'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Landing page — no navbar */}
        <Route path="/" element={<Landing />} />

        {/* App pages — with navbar */}
        <Route path="/app" element={<><Navbar /><Dashboard /></>} />
        <Route path="/app/upload" element={<><Navbar /><Upload /></>} />
        <Route path="/app/chat" element={<><Navbar /><Chat /></>} />
        <Route path="/app/goals" element={<><Navbar /><Goals /></>} />
      </Routes>
    </BrowserRouter>
  )
}
