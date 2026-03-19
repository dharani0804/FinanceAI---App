import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, CartesianGrid
} from 'recharts'

const COLORS = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#f97316']

const S = {
  page: { padding:'32px 48px', maxWidth:1400, margin:'0 auto', fontFamily:'Inter, sans-serif' },
  card: { background:'#12121e', border:'1px solid #1e1e2e', borderRadius:16, padding:'20px 24px' },
  sectionTitle: { color:'#fff', fontSize:14, fontWeight:700, marginBottom:4 },
  sectionSub: { color:'#4a4a6a', fontSize:12, marginBottom:20 },
  tooltip: { background:'#1a1a2e', border:'none', borderRadius:10, color:'#fff', fontSize:12, padding:'8px 12px' },
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={S.tooltip}>
      <p style={{color:'#8888aa', marginBottom:4, fontSize:11}}>{label}</p>
      {payload.map((p,i) => (
        <p key={i} style={{color:p.color, fontWeight:600}}>{p.name}: ${p.value?.toFixed(2)}</p>
      ))}
    </div>
  )
}

const InsightCard = ({ icon, title, desc, color }) => (
  <div style={{...S.card, borderLeft:`3px solid ${color}`, padding:'14px 18px', display:'flex', gap:14, alignItems:'flex-start'}}>
    <span style={{fontSize:20, flexShrink:0}}>{icon}</span>
    <div>
      <p style={{color:'#fff', fontSize:13, fontWeight:600, marginBottom:4}}>{title}</p>
      <p style={{color:'#6a6a8a', fontSize:12, lineHeight:1.5}}>{desc}</p>
    </div>
  </div>
)

function getDateRange(view, customStart, customEnd) {
  const now = new Date()
  if (view === 'month') {
    const start = new Date(now.getFullYear(), now.getMonth(), 1)
    const end = new Date(now.getFullYear(), now.getMonth()+1, 0)
    return { start, end }
  }
  if (view === 'lastmonth') {
    const start = new Date(now.getFullYear(), now.getMonth()-1, 1)
    const end = new Date(now.getFullYear(), now.getMonth(), 0)
    return { start, end }
  }
  if (view === '3months') {
    const start = new Date(now.getFullYear(), now.getMonth()-2, 1)
    const end = new Date(now.getFullYear(), now.getMonth()+1, 0)
    return { start, end }
  }
  if (view === 'ytd') {
    const start = new Date(now.getFullYear(), 0, 1)
    const end = now
    return { start, end }
  }
  if (view === 'custom' && customStart && customEnd) {
    return { start: new Date(customStart), end: new Date(customEnd) }
  }
  // all time
  return { start: new Date('2000-01-01'), end: new Date('2099-12-31') }
}

export default function Dashboard() {
  const [transactions, setTransactions] = useState([])
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState('month')
  const [customStart, setCustomStart] = useState('')
  const [customEnd, setCustomEnd] = useState('')

  useEffect(() => {
    Promise.all([
      fetch('http://127.0.0.1:8000/transactions').then(r=>r.json()),
      fetch('http://127.0.0.1:8000/profile').then(r=>r.json()),
    ]).then(([txs, prof]) => {
      setTransactions(txs)
      setProfile(prof)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{display:'flex', alignItems:'center', justifyContent:'center', height:400, fontFamily:'Inter,sans-serif'}}>
      <p style={{color:'#4a4a6a'}}>Loading your finances...</p>
    </div>
  )

  if (transactions.length === 0) return (
    <div style={{display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:400, fontFamily:'Inter,sans-serif', gap:16}}>
      <p style={{fontSize:40}}>📊</p>
      <p style={{color:'#fff', fontSize:18, fontWeight:700}}>No transactions yet</p>
      <p style={{color:'#6a6a8a', fontSize:14}}>Upload a bank statement to see your dashboard</p>
      <Link to="/app/upload" style={{background:'#2563eb', color:'#fff', padding:'10px 24px', borderRadius:12, textDecoration:'none', fontWeight:600, fontSize:14}}>
        Upload Statement →
      </Link>
    </div>
  )

  // Date range filter
  const { start, end } = getDateRange(view, customStart, customEnd)
  const filtered = transactions.filter(t => {
    if (!t.date) return false
    const d = new Date(t.date)
    return d >= start && d <= end
  })

  // Previous period for comparison
  const periodMs = end - start
  const prevStart = new Date(start - periodMs)
  const prevEnd = new Date(start - 1)
  const prevFiltered = transactions.filter(t => {
    if (!t.date) return false
    const d = new Date(t.date)
    return d >= prevStart && d <= prevEnd
  })

  const expenses = filtered.filter(t=>t.amount<0)
  const income = filtered.filter(t=>t.amount>0)
  const totalSpend = expenses.reduce((s,t)=>s+Math.abs(t.amount),0)
  const totalIncome = profile?.monthly_income
    ? profile.monthly_income * (view==='3months'?3:view==='ytd'?new Date().getMonth()+1:1)
    : income.reduce((s,t)=>s+t.amount,0)
  const saved = totalIncome - totalSpend
  const days = Math.max(1, Math.ceil((end-start)/(1000*60*60*24)))
  const avgDaily = totalSpend / days

  const prevSpend = prevFiltered.filter(t=>t.amount<0).reduce((s,t)=>s+Math.abs(t.amount),0)
  const spendChange = prevSpend > 0 ? ((totalSpend-prevSpend)/prevSpend*100).toFixed(0) : 0
  const spendUp = totalSpend > prevSpend

  // Monthly budget from profile
  const monthlyBudget = profile?.monthly_budget || 0
  const periodBudget = monthlyBudget * (view==='3months'?3:view==='ytd'?new Date().getMonth()+1:1)
  const budgetUsed = periodBudget > 0 ? (totalSpend/periodBudget*100).toFixed(0) : 0

  // Category breakdown
  const catMap = expenses.reduce((acc,t)=>{
    const cat = t.category||'Other'
    acc[cat] = (acc[cat]||0)+Math.abs(t.amount)
    return acc
  },{})
  const topCats = Object.entries(catMap)
    .map(([name,value])=>({name, value:parseFloat(value.toFixed(2))}))
    .sort((a,b)=>b.value-a.value)
    .slice(0,5)

  const donutData = Object.entries(catMap)
    .map(([name,value])=>({name, value:parseFloat(value.toFixed(2))}))
    .sort((a,b)=>b.value-a.value)
    .slice(0,6)

  // Monthly trend
  const monthlyMap = {}
  transactions.forEach(t=>{
    if(!t.date) return
    const m = t.date.slice(0,7)
    if(!monthlyMap[m]) monthlyMap[m]={month:m,Spend:0,Income:0}
    if(t.amount<0) monthlyMap[m].Spend+=Math.abs(t.amount)
    else monthlyMap[m].Income+=t.amount
  })
  const monthlyTrend = Object.values(monthlyMap)
    .sort((a,b)=>a.month.localeCompare(b.month))
    .slice(-6)
    .map(d=>({...d, Spend:parseFloat(d.Spend.toFixed(2)), Income:parseFloat(d.Income.toFixed(2))}))

  // Budget vs actual chart
  const budgetData = topCats.map(c=>({
    name: c.name.split(' ')[0],
    Actual: c.value,
    Budget: periodBudget > 0
      ? parseFloat(((periodBudget/topCats.length)*1.1).toFixed(2))
      : parseFloat((c.value*1.2).toFixed(2)),
  }))

  // AI insights
  const topCat = topCats[0]
  const largestTx = expenses.sort((a,b)=>a.amount-b.amount)[0]
  const insights = [
    {
      icon: spendUp?'📈':'📉',
      color: spendUp?'#ef4444':'#10b981',
      title: spendUp?`Spending up ${Math.abs(spendChange)}% vs previous period`:`Spending down ${Math.abs(spendChange)}% vs previous period`,
      desc: `$${totalSpend.toFixed(2)} spent vs $${prevSpend.toFixed(2)} previous period.`
    },
    topCat && {
      icon:'🏆', color:'#f59e0b',
      title:`${topCat.name} is your #1 category`,
      desc:`$${topCat.value.toFixed(2)} spent — ${(topCat.value/totalSpend*100).toFixed(0)}% of total spend.`
    },
    largestTx && {
      icon:'💸', color:'#8b5cf6',
      title:`Largest expense: ${largestTx.description}`,
      desc:`$${Math.abs(largestTx.amount).toFixed(2)} on ${largestTx.date} · ${largestTx.category}`
    },
    periodBudget > 0 && {
      icon: budgetUsed > 90?'⚠️':budgetUsed>70?'🟡':'✅',
      color: budgetUsed>90?'#ef4444':budgetUsed>70?'#f59e0b':'#10b981',
      title: `${budgetUsed}% of budget used`,
      desc: budgetUsed>100?`You're $${(totalSpend-periodBudget).toFixed(2)} over budget this period.`:`$${(periodBudget-totalSpend).toFixed(2)} remaining in your budget.`
    },
    {
      icon:'📅', color:'#3b82f6',
      title:`Avg daily spend: $${avgDaily.toFixed(2)}`,
      desc:`Across ${days} days in the selected period.`
    },
  ].filter(Boolean).slice(0,5)

  const viewOptions = [
    {key:'month',     label:'This Month'},
    {key:'lastmonth', label:'Last Month'},
    {key:'3months',   label:'3 Months'},
    {key:'ytd',       label:'YTD'},
    {key:'custom',    label:'Custom'},
    {key:'all',       label:'All Time'},
  ]

  const periodLabel = view==='month'?'This Month':view==='lastmonth'?'Last Month':view==='3months'?'Last 3 Months':view==='ytd'?'Year to Date':view==='custom'?`${customStart} → ${customEnd}`:'All Time'

  return (
    <div style={S.page}>

      {/* HEADER */}
      <div style={{display:'flex', alignItems:'flex-start', justifyContent:'space-between', marginBottom:28, flexWrap:'wrap', gap:16}}>
        <div>
          <h1 style={{fontSize:24, fontWeight:800, color:'#fff', letterSpacing:'-0.5px', marginBottom:4}}>Dashboard</h1>
          <p style={{color:'#6a6a8a', fontSize:13}}>{periodLabel} · {filtered.length} transactions</p>
        </div>

        {/* Date filter */}
        <div style={{display:'flex', flexDirection:'column', gap:8, alignItems:'flex-end'}}>
          <div style={{display:'flex', gap:4, background:'#12121e', border:'1px solid #1e1e2e', borderRadius:12, padding:4}}>
            {viewOptions.map(({key,label})=>(
              <button key={key} onClick={()=>setView(key)} style={{
                padding:'6px 14px', borderRadius:8, border:'none', cursor:'pointer', fontSize:12, fontWeight:600,
                background: view===key?'#2563eb':'transparent',
                color: view===key?'#fff':'#6a6a8a',
                fontFamily:'Inter,sans-serif',
              }}>{label}</button>
            ))}
          </div>

          {/* Custom date range */}
          {view === 'custom' && (
            <div style={{display:'flex', gap:8, alignItems:'center'}}>
              <input type="date" value={customStart} onChange={e=>setCustomStart(e.target.value)}
                style={{background:'#12121e', border:'1px solid #2a2a3a', borderRadius:8, padding:'6px 12px', color:'#fff', fontSize:12, outline:'none', fontFamily:'Inter,sans-serif'}}/>
              <span style={{color:'#4a4a6a', fontSize:12}}>to</span>
              <input type="date" value={customEnd} onChange={e=>setCustomEnd(e.target.value)}
                style={{background:'#12121e', border:'1px solid #2a2a3a', borderRadius:8, padding:'6px 12px', color:'#fff', fontSize:12, outline:'none', fontFamily:'Inter,sans-serif'}}/>
            </div>
          )}
        </div>
      </div>

      {/* SCORECARDS */}
      <div style={{display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:16, marginBottom:20}}>
        {[
          { label:"Total Spend",    value:`$${totalSpend.toFixed(2)}`,  sub:`${spendUp?'↑':'↓'} ${Math.abs(spendChange)}% vs prev period`, color:spendUp?'#ef4444':'#10b981', icon:'💳', bg:spendUp?'rgba(239,68,68,0.08)':'rgba(16,185,129,0.08)' },
          { label:"Income",         value:`$${totalIncome.toFixed(2)}`,  sub: profile?.monthly_income?'From your profile':'From transactions', color:'#10b981', icon:'💵', bg:'rgba(16,185,129,0.08)' },
          { label:"Saved",          value:`$${Math.max(0,saved).toFixed(2)}`, sub:totalIncome>0?`${((Math.max(0,saved)/totalIncome)*100).toFixed(0)}% savings rate`:'—', color:'#3b82f6', icon:'🏦', bg:'rgba(59,130,246,0.08)' },
          { label:"Avg Daily Spend",value:`$${avgDaily.toFixed(2)}`,    sub:`Over ${days} days`, color:'#8b5cf6', icon:'📅', bg:'rgba(139,92,246,0.08)' },
        ].map((s,i)=>(
          <div key={i} style={S.card}>
            <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:16}}>
              <div style={{width:40, height:40, borderRadius:12, background:s.bg, display:'flex', alignItems:'center', justifyContent:'center', fontSize:18}}>{s.icon}</div>
            </div>
            <p style={{color:'#6a6a8a', fontSize:11, fontWeight:600, letterSpacing:1.5, textTransform:'uppercase', marginBottom:8}}>{s.label}</p>
            <p style={{fontSize:26, fontWeight:800, color:s.color, marginBottom:4, letterSpacing:'-0.5px'}}>{s.value}</p>
            <p style={{color:'#4a4a6a', fontSize:12}}>{s.sub}</p>
          </div>
        ))}
      </div>

      {/* Budget used progress bar */}
      {periodBudget > 0 && (
        <div style={{...S.card, marginBottom:20}}>
          <div style={{display:'flex', justifyContent:'space-between', marginBottom:10}}>
            <p style={{color:'#fff', fontWeight:600, fontSize:13}}>Budget Used</p>
            <p style={{color: budgetUsed>100?'#ef4444':budgetUsed>80?'#f59e0b':'#10b981', fontWeight:700, fontSize:13}}>{budgetUsed}% of ${periodBudget.toLocaleString()}</p>
          </div>
          <div style={{background:'#1e1e2e', borderRadius:99, height:10, overflow:'hidden'}}>
            <div style={{
              height:'100%', borderRadius:99,
              background: budgetUsed>100?'#ef4444':budgetUsed>80?'#f59e0b':'#10b981',
              width:`${Math.min(100,budgetUsed)}%`,
              transition:'width 0.6s ease'
            }}/>
          </div>
          <div style={{display:'flex', justifyContent:'space-between', marginTop:6}}>
            <p style={{color:'#4a4a6a', fontSize:11}}>$0</p>
            <p style={{color:'#4a4a6a', fontSize:11}}>${totalSpend.toFixed(0)} spent · ${Math.max(0,periodBudget-totalSpend).toFixed(0)} remaining</p>
            <p style={{color:'#4a4a6a', fontSize:11}}>${periodBudget.toLocaleString()}</p>
          </div>
        </div>
      )}

      {/* BUDGET VS ACTUAL + TOP CATEGORIES */}
      <div style={{display:'grid', gridTemplateColumns:'1.2fr 1fr', gap:16, marginBottom:20}}>
        <div style={S.card}>
          <p style={S.sectionTitle}>Budget vs Actual</p>
          <p style={S.sectionSub}>Estimated budget vs what you spent</p>
          {budgetData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={budgetData} barGap={4} barCategoryGap="25%">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" vertical={false}/>
                <XAxis dataKey="name" tick={{fill:'#4a4a6a',fontSize:11}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:'#4a4a6a',fontSize:11}} axisLine={false} tickLine={false} tickFormatter={v=>`$${v}`}/>
                <Tooltip content={<CustomTooltip/>} cursor={{fill:'rgba(255,255,255,0.02)'}}/>
                <Bar dataKey="Budget" fill="#2a2a3a" radius={[4,4,0,0]} name="Budget"/>
                <Bar dataKey="Actual" fill="#3b82f6" radius={[4,4,0,0]} name="Actual"/>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{height:220, display:'flex', alignItems:'center', justifyContent:'center'}}>
              <p style={{color:'#4a4a6a', fontSize:13}}>No spending data for this period</p>
            </div>
          )}
        </div>

        <div style={S.card}>
          <p style={S.sectionTitle}>Top 5 Spending Categories</p>
          <p style={S.sectionSub}>Where your money went</p>
          {topCats.length > 0 ? (
            <div style={{marginTop:8}}>
              {topCats.map((cat,i)=>(
                <div key={i} style={{marginBottom:14}}>
                  <div style={{display:'flex', justifyContent:'space-between', marginBottom:5}}>
                    <span style={{color:'#c0c0d8', fontSize:13, fontWeight:500}}>{cat.name}</span>
                    <span style={{color:'#fff', fontSize:13, fontWeight:700}}>${cat.value.toFixed(2)}</span>
                  </div>
                  <div style={{background:'#1e1e2e', borderRadius:99, height:6, overflow:'hidden'}}>
                    <div style={{height:'100%', borderRadius:99, background:COLORS[i%COLORS.length], width:`${(cat.value/topCats[0].value*100).toFixed(0)}%`, transition:'width 0.6s ease'}}/>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p style={{color:'#4a4a6a', fontSize:13, marginTop:20}}>No expense data for this period</p>
          )}
        </div>
      </div>

      {/* TREND + DONUT */}
      <div style={{display:'grid', gridTemplateColumns:'1.5fr 1fr', gap:16, marginBottom:20}}>
        <div style={S.card}>
          <p style={S.sectionTitle}>Monthly Spend Trend</p>
          <p style={S.sectionSub}>Income vs spending over time</p>
          {monthlyTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={monthlyTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" vertical={false}/>
                <XAxis dataKey="month" tick={{fill:'#4a4a6a',fontSize:11}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:'#4a4a6a',fontSize:11}} axisLine={false} tickLine={false} tickFormatter={v=>`$${v}`}/>
                <Tooltip content={<CustomTooltip/>}/>
                <Line type="monotone" dataKey="Income" stroke="#10b981" strokeWidth={2.5} dot={{fill:'#10b981',r:4}} name="Income"/>
                <Line type="monotone" dataKey="Spend" stroke="#3b82f6" strokeWidth={2.5} dot={{fill:'#3b82f6',r:4}} name="Spend"/>
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{height:220, display:'flex', alignItems:'center', justifyContent:'center'}}>
              <p style={{color:'#4a4a6a', fontSize:13}}>Not enough data for trend</p>
            </div>
          )}
        </div>

        <div style={S.card}>
          <p style={S.sectionTitle}>Spending Breakdown</p>
          <p style={S.sectionSub}>By category</p>
          {donutData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={150}>
                <PieChart>
                  <Pie data={donutData} cx="50%" cy="50%" innerRadius={42} outerRadius={68} dataKey="value" paddingAngle={3} strokeWidth={0}>
                    {donutData.map((_,i)=><Cell key={i} fill={COLORS[i%COLORS.length]}/>)}
                  </Pie>
                  <Tooltip formatter={v=>[`$${v.toFixed(2)}`,'Amount']} contentStyle={{background:'#1a1a2e',border:'none',borderRadius:8,color:'#fff',fontSize:12}}/>
                </PieChart>
              </ResponsiveContainer>
              <div style={{marginTop:8}}>
                {donutData.map((d,i)=>(
                  <div key={i} style={{display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:6}}>
                    <div style={{display:'flex', alignItems:'center', gap:8}}>
                      <div style={{width:8, height:8, borderRadius:'50%', background:COLORS[i%COLORS.length]}}/>
                      <span style={{color:'#8888aa', fontSize:12}}>{d.name}</span>
                    </div>
                    <span style={{color:'#fff', fontSize:12, fontWeight:600}}>${d.value.toFixed(0)}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p style={{color:'#4a4a6a', fontSize:13, marginTop:20}}>No data for this period</p>
          )}
        </div>
      </div>

      {/* AI INSIGHTS */}
      <div style={{...S.card, marginBottom:20}}>
        <p style={{...S.sectionTitle, marginBottom:4}}>🤖 AI Insights</p>
        <p style={{...S.sectionSub, marginBottom:16}}>Smart observations about your finances</p>
        <div style={{display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:12}}>
          {insights.map((insight,i)=>(
            <InsightCard key={i} {...insight}/>
          ))}
        </div>
      </div>

      {/* TRANSACTIONS */}
      <div style={S.card}>
        <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16}}>
          <div>
            <p style={S.sectionTitle}>Transactions</p>
            <p style={{color:'#4a4a6a', fontSize:12}}>{filtered.length} records in selected period</p>
          </div>
        </div>
        <table style={{width:'100%', borderCollapse:'collapse'}}>
          <thead>
            <tr style={{borderBottom:'1px solid #1e1e2e'}}>
              {['Date','Description','Category','Bank','Amount'].map((h,i)=>(
                <th key={h} style={{padding:'10px 14px', color:'#4a4a6a', fontSize:11, fontWeight:600, textTransform:'uppercase', letterSpacing:1, textAlign:i===4?'right':'left', borderBottom:'1px solid #1e1e2e'}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0,20).map((t,i)=>(
              <tr key={i} style={{borderBottom:'1px solid #1a1a2a'}}>
                <td style={{padding:'11px 14px', color:'#6a6a8a', fontSize:12}}>{t.date}</td>
                <td style={{padding:'11px 14px', color:'#c0c0d8', fontSize:13, fontWeight:500}}>{t.description}</td>
                <td style={{padding:'11px 14px'}}>
                  <span style={{background:'#1e1e2e', color:'#8888aa', fontSize:11, padding:'3px 10px', borderRadius:6}}>{t.category}</span>
                </td>
                <td style={{padding:'11px 14px', color:'#4a4a6a', fontSize:12}}>{t.bank_source}</td>
                <td style={{padding:'11px 14px', textAlign:'right', fontWeight:700, fontSize:13, color:t.amount>=0?'#10b981':'#ef4444'}}>
                  {t.amount>=0?'+':'-'}${Math.abs(t.amount).toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  )
}
