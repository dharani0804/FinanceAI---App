import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, Upload, MessageCircle, Target } from 'lucide-react'

const links = [
  { to: '/app/upload', label: 'Upload',    icon: Upload },
  { to: '/app/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/app/chat',   label: 'AI Chat',   icon: MessageCircle },
  { to: '/app/goals',  label: 'Goals',     icon: Target },
]

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 48px', height: 64,
      background: 'rgba(10,10,15,0.97)', backdropFilter: 'blur(20px)',
      borderBottom: '1px solid #1e1e2e', position: 'sticky', top: 0, zIndex: 50,
      fontFamily: 'Inter, sans-serif',
    }}>

      <Link to="/" style={{display:'flex', alignItems:'center', gap:10, textDecoration:'none', minWidth:140}}>
        <div style={{width:34, height:34, borderRadius:10, background:'linear-gradient(135deg,#3b82f6,#1d4ed8)', display:'flex', alignItems:'center', justifyContent:'center', color:'#fff', fontWeight:800, fontSize:15, boxShadow:'0 4px 14px rgba(59,130,246,0.3)'}}>F</div>
        <span style={{fontWeight:700, fontSize:16, color:'#fff'}}>FinanceAI</span>
      </Link>

      <div style={{display:'flex', alignItems:'center', gap:4, background:'#12121e', border:'1px solid #1e1e2e', borderRadius:16, padding:4}}>
        {links.map(({ to, label, icon: Icon }) => {
          const active = pathname === to
          return (
            <Link key={to} to={to} style={{
              display:'flex', alignItems:'center', gap:8,
              padding:'8px 20px', borderRadius:12,
              fontSize:14, fontWeight:600, textDecoration:'none', transition:'all 0.2s',
              background: active ? '#2563eb' : 'transparent',
              color: active ? '#fff' : '#6a6a8a',
              boxShadow: active ? '0 2px 12px rgba(37,99,235,0.3)' : 'none',
            }}>
              <Icon size={15} strokeWidth={active ? 2.5 : 2} />
              <span>{label}</span>
            </Link>
          )
        })}
      </div>

      <div style={{display:'flex', alignItems:'center', gap:8, background:'#12121e', border:'1px solid #1e1e2e', borderRadius:12, padding:'6px 14px', minWidth:140, justifyContent:'flex-end'}}>
        <div style={{width:26, height:26, borderRadius:'50%', background:'linear-gradient(135deg,#3b82f6,#8b5cf6)', display:'flex', alignItems:'center', justifyContent:'center', color:'#fff', fontSize:11, fontWeight:800}}>D</div>
        <span style={{color:'#8888aa', fontSize:13, fontWeight:500}}>Dharani</span>
      </div>

    </nav>
  )
}
