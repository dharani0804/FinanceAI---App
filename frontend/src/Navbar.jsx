import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, Upload, MessageCircle, Target } from 'lucide-react'

const links = [
  { to: '/app',        label: 'Dashboard', icon: LayoutDashboard },
  { to: '/app/upload', label: 'Upload',    icon: Upload },
  { to: '/app/chat',   label: 'AI Chat',   icon: MessageCircle },
  { to: '/app/goals',  label: 'Goals',     icon: Target },
]

export default function Navbar() {
  const { pathname } = useLocation()
  return (
    <nav style={{borderBottom:'1px solid #1e1e2e', backdropFilter:'blur(12px)'}}
      className="bg-[#0a0a0f]/95 px-10 sticky top-0 z-50 flex items-center justify-between h-16">
      <Link to="/" className="flex items-center gap-2.5 min-w-[140px]">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center text-white text-sm font-bold shadow-lg shadow-blue-500/25">F</div>
        <span className="font-semibold text-white text-base tracking-tight">FinanceAI</span>
      </Link>
      <div className="flex items-center gap-1 bg-[#12121e] border border-[#1e1e2e] rounded-2xl px-2 py-1.5">
        {links.map(({ to, label, icon: Icon }) => {
          const active = pathname === to
          return (
            <Link key={to} to={to}
              className={`flex items-center gap-2.5 px-5 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                active ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-[#6a6a8a] hover:text-white hover:bg-white/5'
              }`}>
              <Icon size={15} strokeWidth={active ? 2.5 : 2} />
              <span>{label}</span>
            </Link>
          )
        })}
      </div>
      <div className="flex items-center gap-3 min-w-[140px] justify-end">
        <div style={{border:'1px solid #1e1e2e'}} className="flex items-center gap-2 bg-[#12121e] px-3 py-1.5 rounded-xl">
          <div className="w-6 h-6 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold">D</div>
          <span className="text-[#8888aa] text-xs font-medium">Dharani</span>
        </div>
      </div>
    </nav>
  )
}
