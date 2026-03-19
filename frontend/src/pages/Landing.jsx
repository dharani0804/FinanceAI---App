import { Link } from 'react-router-dom'

const scrollTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })

const S = {
  page: { minHeight:'100vh', background:'#0a0a0f', color:'#fff', fontFamily:'Inter, sans-serif' },
  nav: { display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0 64px', height:72, background:'rgba(10,10,15,0.95)', backdropFilter:'blur(20px)', borderBottom:'1px solid #1e1e2e', position:'sticky', top:0, zIndex:50 },
  logo: { display:'flex', alignItems:'center', gap:10 },
  logoBox: { width:36, height:36, borderRadius:10, background:'linear-gradient(135deg,#3b82f6,#1d4ed8)', display:'flex', alignItems:'center', justifyContent:'center', color:'#fff', fontWeight:800, fontSize:16 },
  logoText: { fontWeight:800, fontSize:18, color:'#fff' },
  navLinks: { display:'flex', alignItems:'center', gap:4 },
  navBtn: { background:'none', border:'none', color:'#8888aa', fontSize:15, fontWeight:500, padding:'8px 18px', borderRadius:10, cursor:'pointer', transition:'all 0.2s' },
  navRight: { display:'flex', alignItems:'center', gap:12 },
  signIn: { color:'#8888aa', fontSize:15, fontWeight:500, padding:'8px 18px', borderRadius:10, textDecoration:'none', border:'none', background:'none', cursor:'pointer' },
  ctaBtn: { background:'#2563eb', color:'#fff', fontSize:15, fontWeight:700, padding:'10px 24px', borderRadius:12, textDecoration:'none', boxShadow:'0 4px 20px rgba(37,99,235,0.3)' },
}

export default function Landing() {
  return (
    <div style={S.page}>

      {/* NAV */}
      <nav style={S.nav}>
        <div style={S.logo}>
          <div style={S.logoBox}>F</div>
          <span style={S.logoText}>FinanceAI</span>
        </div>
        <div style={S.navLinks}>
          {[['Features','features'],['How It Works','howitworks'],['About','about']].map(([label,id])=>(
            <button key={id} onClick={()=>scrollTo(id)} style={S.navBtn}
              onMouseEnter={e=>e.target.style.color='#fff'}
              onMouseLeave={e=>e.target.style.color='#8888aa'}>
              {label}
            </button>
          ))}
        </div>
        <div style={S.navRight}>
          <Link to="/app" style={S.signIn}>Sign In</Link>
          <Link to="/app" style={S.ctaBtn}>Get Started Free →</Link>
        </div>
      </nav>

      {/* HERO */}
      <section style={{textAlign:'center', padding:'100px 64px 80px', maxWidth:900, margin:'0 auto', position:'relative'}}>
        <div style={{position:'absolute', top:0, left:'50%', transform:'translateX(-50%)', width:600, height:300, background:'radial-gradient(ellipse,rgba(59,130,246,0.12),transparent 70%)', pointerEvents:'none'}} />
        <div style={{display:'inline-flex', alignItems:'center', gap:8, background:'rgba(59,130,246,0.1)', border:'1px solid rgba(59,130,246,0.25)', color:'#60a5fa', fontSize:13, fontWeight:600, padding:'6px 16px', borderRadius:99, marginBottom:32}}>
          ⚡ AI-Powered Personal Finance
        </div>
        <h1 style={{fontSize:64, fontWeight:900, lineHeight:1.08, marginBottom:24, letterSpacing:'-2px'}}>
          Finally understand<br/>
          <span style={{background:'linear-gradient(135deg,#60a5fa,#a78bfa)', WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent'}}>
            where your money goes
          </span>
        </h1>
        <p style={{color:'#8888aa', fontSize:20, lineHeight:1.7, marginBottom:40, maxWidth:560, margin:'0 auto 40px'}}>
          Upload any bank statement. Get an instant AI breakdown of your spending. Chat with an assistant that knows your finances.
        </p>
        <div style={{display:'flex', alignItems:'center', justifyContent:'center', gap:16, marginBottom:20}}>
          <Link to="/app" style={{display:'inline-flex', alignItems:'center', gap:8, background:'#2563eb', color:'#fff', fontWeight:700, fontSize:16, padding:'14px 36px', borderRadius:14, textDecoration:'none', boxShadow:'0 8px 32px rgba(37,99,235,0.35)'}}>
            Get Started Free →
          </Link>
          <button onClick={()=>scrollTo('features')} style={{display:'inline-flex', alignItems:'center', gap:8, background:'none', color:'#8888aa', fontWeight:600, fontSize:16, padding:'14px 32px', borderRadius:14, border:'1px solid #2a2a3a', cursor:'pointer'}}>
            See Features
          </button>
        </div>
        <p style={{color:'#4a4a6a', fontSize:13}}>Free forever · No credit card · Your data stays on your device</p>
      </section>

      {/* PROBLEM STRIP */}
      <section style={{padding:'0 64px 80px', maxWidth:1100, margin:'0 auto'}}>
        <div style={{background:'#12121e', borderRadius:24, padding:'48px 56px', border:'1px solid #1e1e2e'}}>
          <p style={{textAlign:'center', color:'#4a4a6a', fontSize:12, fontWeight:600, letterSpacing:3, textTransform:'uppercase', marginBottom:40}}>Sound familiar?</p>
          <div style={{display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:32}}>
            {[
              ['🏦','Bank apps show transactions but zero insight'],
              ['📊','Spreadsheets need manual input every single time'],
              ['🤔','No idea why you overspent last month'],
              ['🔀','Multiple accounts, total confusion'],
            ].map(([emoji,text],i)=>(
              <div key={i} style={{textAlign:'center'}}>
                <div style={{fontSize:36, marginBottom:16}}>{emoji}</div>
                <p style={{color:'#6a6a8a', fontSize:14, lineHeight:1.6}}>{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" style={{padding:'80px 64px', maxWidth:1200, margin:'0 auto', scrollMarginTop:80}}>
        <div style={{textAlign:'center', marginBottom:56}}>
          <p style={{color:'#4a4a6a', fontSize:12, fontWeight:600, letterSpacing:3, textTransform:'uppercase', marginBottom:12}}>Features</p>
          <h2 style={{fontSize:40, fontWeight:800, color:'#fff', marginBottom:16, letterSpacing:'-1px'}}>Everything you need to take control</h2>
          <p style={{color:'#6a6a8a', fontSize:16}}>Built for everyday people — not accountants.</p>
        </div>
        <div style={{display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:20}}>
          {[
            ['📎','Upload Any Statement',    '#3b82f6','Import CSV or PDF from any bank worldwide. Instant parsing, no manual entry, no credentials required.'],
            ['📊','Visual Dashboard',         '#8b5cf6','Scorecards, bar charts and category breakdowns appear the second you upload.'],
            ['💬','AI Chat Assistant',        '#10b981','Ask Claude AI anything — why you overspent, where to save. Real answers from your real data.'],
            ['🎯','Goals & Recommendations', '#f59e0b','Set savings targets with deadlines. Get AI advice tailored to your actual spending patterns.'],
            ['🔒','Private by Default',       '#ef4444','No cloud database. Your financial data never leaves your device.'],
            ['🌍','Multi-Currency Support',   '#06b6d4','Works with USD, EUR, GBP, INR, AUD and 150+ currencies. Built for global users.'],
          ].map(([emoji,title,color,desc],i)=>(
            <div key={i} style={{background:'#12121e', borderRadius:20, padding:'28px', border:'1px solid #1e1e2e', transition:'transform 0.2s'}}
              onMouseEnter={e=>e.currentTarget.style.transform='translateY(-4px)'}
              onMouseLeave={e=>e.currentTarget.style.transform='translateY(0)'}>
              <div style={{width:48, height:48, borderRadius:14, background:`${color}18`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:22, marginBottom:20}}>{emoji}</div>
              <h3 style={{color:'#fff', fontWeight:700, fontSize:15, marginBottom:10}}>{title}</h3>
              <p style={{color:'#6a6a8a', fontSize:13, lineHeight:1.7}}>{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="howitworks" style={{padding:'80px 64px', maxWidth:900, margin:'0 auto', scrollMarginTop:80}}>
        <div style={{textAlign:'center', marginBottom:56}}>
          <p style={{color:'#4a4a6a', fontSize:12, fontWeight:600, letterSpacing:3, textTransform:'uppercase', marginBottom:12}}>How It Works</p>
          <h2 style={{fontSize:40, fontWeight:800, color:'#fff', letterSpacing:'-1px'}}>Up and running in 3 steps</h2>
        </div>
        <div style={{display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:40, position:'relative'}}>
          <div style={{position:'absolute', top:36, left:'20%', right:'20%', height:1, background:'linear-gradient(90deg,#3b82f6,#8b5cf6,#10b981)', opacity:0.4}} />
          {[
            ['01','#3b82f6','Upload your statement','Download a CSV or PDF from your bank app and upload it. Takes 30 seconds.'],
            ['02','#8b5cf6','Get instant insights','Dashboard populates automatically — spending by category, monthly trends, key metrics.'],
            ['03','#10b981','Chat with your AI','Ask anything about your money. Get specific data-driven answers from Claude AI.'],
          ].map(([step,color,title,desc],i)=>(
            <div key={i} style={{textAlign:'center', position:'relative', zIndex:1}}>
              <div style={{width:64, height:64, borderRadius:18, background:`${color}15`, border:`1px solid ${color}40`, display:'flex', alignItems:'center', justifyContent:'center', margin:'0 auto 24px', fontSize:20, fontWeight:900, color}}>{step}</div>
              <h3 style={{color:'#fff', fontWeight:700, fontSize:15, marginBottom:10}}>{title}</h3>
              <p style={{color:'#6a6a8a', fontSize:13, lineHeight:1.7}}>{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* PRICING */}

      {/* ABOUT */}
      <section id="about" style={{padding:'80px 64px', maxWidth:900, margin:'0 auto', scrollMarginTop:80}}>
        <div style={{background:'#12121e', borderRadius:28, padding:'64px', textAlign:'center', border:'1px solid #1e1e2e'}}>
          <p style={{color:'#4a4a6a', fontSize:12, fontWeight:600, letterSpacing:3, textTransform:'uppercase', marginBottom:16}}>About FinanceAI</p>
          <h2 style={{fontSize:32, fontWeight:800, color:'#fff', marginBottom:20, letterSpacing:'-1px'}}>Built for real people, not spreadsheet experts</h2>
          <p style={{color:'#6a6a8a', fontSize:15, lineHeight:1.8, maxWidth:600, margin:'0 auto 48px'}}>
            FinanceAI was built to solve a simple problem — most people don't know where their money goes, and existing tools are either too complex, too manual, or require handing over bank credentials. We built a privacy-first, AI-powered alternative that works with any bank, in any currency, with zero manual input.
          </p>
          <div style={{display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:32}}>
            {[['150+','Currencies Supported'],['$0','To Get Started'],['100%','Private & Local']].map(([val,label],i)=>(
              <div key={i}>
                <p style={{fontSize:36, fontWeight:900, color:'#60a5fa', marginBottom:6}}>{val}</p>
                <p style={{color:'#6a6a8a', fontSize:14}}>{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA BANNER */}
      <section style={{padding:'0 64px 80px', maxWidth:800, margin:'0 auto'}}>
        <div style={{background:'linear-gradient(135deg,rgba(59,130,246,0.12),rgba(139,92,246,0.12))', borderRadius:28, padding:'72px 64px', textAlign:'center', border:'1px solid rgba(99,102,241,0.25)', position:'relative', overflow:'hidden'}}>
          <div style={{position:'absolute', top:0, left:'50%', transform:'translateX(-50%)', width:300, height:100, background:'rgba(59,130,246,0.15)', borderRadius:'50%', filter:'blur(40px)'}} />
          <h2 style={{fontSize:38, fontWeight:900, color:'#fff', marginBottom:16, letterSpacing:'-1px', position:'relative'}}>Ready to know your money?</h2>
          <p style={{color:'#8888aa', fontSize:15, lineHeight:1.7, marginBottom:36, maxWidth:400, margin:'0 auto 36px', position:'relative'}}>
            Stop guessing. Start understanding exactly where every dollar goes.
          </p>
          <Link to="/app" style={{display:'inline-flex', alignItems:'center', gap:8, background:'#2563eb', color:'#fff', fontWeight:700, fontSize:16, padding:'14px 40px', borderRadius:14, textDecoration:'none', boxShadow:'0 8px 32px rgba(37,99,235,0.35)', position:'relative'}}>
            Get Started Free →
          </Link>
          <p style={{color:'#4a4a6a', fontSize:13, marginTop:16}}>No credit card required</p>
        </div>
      </section>

      {/* FOOTER */}
      <footer style={{display:'flex', alignItems:'center', justifyContent:'space-between', padding:'24px 64px', borderTop:'1px solid #1e1e2e', maxWidth:1200, margin:'0 auto'}}>
        <div style={{display:'flex', alignItems:'center', gap:8}}>
          <div style={{width:24, height:24, borderRadius:7, background:'#2563eb', display:'flex', alignItems:'center', justifyContent:'center', color:'#fff', fontSize:11, fontWeight:800}}>F</div>
          <span style={{color:'#4a4a6a', fontSize:14, fontWeight:500}}>FinanceAI</span>
        </div>
        <p style={{color:'#4a4a6a', fontSize:13}}>Built with Claude AI · © 2026 FinanceAI</p>
        <div style={{display:'flex', gap:24}}>
          {['Privacy','Terms','Contact'].map(l=>(
            <a key={l} href="#" style={{color:'#4a4a6a', fontSize:13, textDecoration:'none'}}>{l}</a>
          ))}
        </div>
      </footer>

    </div>
  )
}
