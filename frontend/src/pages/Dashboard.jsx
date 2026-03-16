import { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, DollarSign, Activity, ArrowUpRight, ArrowDownRight } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const COLORS = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444','#06b6d4','#f97316']

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{background:'#12121e', border:'1px solid #2a2a3a', borderRadius:10, padding:'10px 14px'}}>
        <p style={{color:'#8888aa', fontSize:11, marginBottom:4}}>{label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{color: p.color, fontSize:13, fontWeight:600}}>
            {p.name}: ${p.value?.toFixed(2)}
          </p>
        ))}
      </div>
    )
  }
  return null
}

const CustomPieTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{background:'#12121e', border:'1px solid #2a2a3a', borderRadius:10, padding:'8px 12px'}}>
        <p style={{color:'#fff', fontSize:13, fontWeight:600}}>{payload[0].name}</p>
        <p style={{color: payload[0].payload.fill, fontSize:12}}>${payload[0].value?.toFixed(2)}</p>
      </div>
    )
  }
  return null
}

export default function Dashboard() {
  const [transactions, setTransactions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('http://127.0.0.1:8000/transactions')
      .then(r => r.json())
      .then(d => { setTransactions(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const expenses  = transactions.filter(t => t.amount < 0)
  const income    = transactions.filter(t => t.amount > 0)
  const totalSpent  = expenses.reduce((s, t) => s + Math.abs(t.amount), 0)
  const totalIncome = income.reduce((s, t) => s + t.amount, 0)
  const balance     = totalIncome - totalSpent
  const savingsRate = totalIncome > 0 ? ((balance / totalIncome) * 100).toFixed(0) : 0

  const categoryData = expenses.reduce((acc, t) => {
    const cat = t.category || 'Other'
    acc[cat] = (acc[cat] || 0) + Math.abs(t.amount)
    return acc
  }, {})
  const pieData = Object.entries(categoryData)
    .map(([name, value]) => ({ name, value: parseFloat(value.toFixed(2)) }))
    .sort((a, b) => b.value - a.value)

  const monthlyData = transactions.reduce((acc, t) => {
    if (!t.date) return acc
    const month = t.date.slice(0, 7)
    if (!acc[month]) acc[month] = { month, Spent: 0, Income: 0 }
    if (t.amount < 0) acc[month].Spent  += Math.abs(t.amount)
    else              acc[month].Income += t.amount
    return acc
  }, {})
  const barData = Object.values(monthlyData)
    .sort((a, b) => a.month.localeCompare(b.month))
    .slice(-6)
    .map(d => ({ ...d, Spent: parseFloat(d.Spent.toFixed(2)), Income: parseFloat(d.Income.toFixed(2)) }))

  const scorecards = [
    {
      label: 'Total Balance',
      value: `$${balance.toFixed(2)}`,
      sub: balance >= 0 ? 'Positive cashflow' : 'Spending exceeds income',
      icon: DollarSign,
      color: '#3b82f6',
      bg: 'rgba(59,130,246,0.08)',
      trend: balance >= 0 ? 'up' : 'down',
    },
    {
      label: 'Total Income',
      value: `$${totalIncome.toFixed(2)}`,
      sub: `${income.length} deposits`,
      icon: TrendingUp,
      color: '#10b981',
      bg: 'rgba(16,185,129,0.08)',
      trend: 'up',
    },
    {
      label: 'Total Spent',
      value: `$${totalSpent.toFixed(2)}`,
      sub: `${expenses.length} transactions`,
      icon: TrendingDown,
      color: '#ef4444',
      bg: 'rgba(239,68,68,0.08)',
      trend: 'down',
    },
    {
      label: 'Savings Rate',
      value: `${savingsRate}%`,
      sub: 'of income saved',
      icon: Activity,
      color: '#8b5cf6',
      bg: 'rgba(139,92,246,0.08)',
      trend: savingsRate > 20 ? 'up' : 'down',
    },
  ]

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <p className="text-[#4a4a6a] text-sm">Loading...</p>
    </div>
  )

  return (
    <div className="p-8 max-w-7xl mx-auto">

      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white mb-1">Financial Overview</h1>
          <p className="text-[#4a4a6a] text-sm">Track your spending, income and savings</p>
        </div>
        {transactions.length === 0 && (
          <Link to="/upload" className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-4 py-2 rounded-lg transition-all">
            + Upload Statement
          </Link>
        )}
      </div>

      {/* Scorecards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        {scorecards.map((s, i) => {
          const Icon = s.icon
          return (
            <div key={i} className="rounded-2xl p-5 relative overflow-hidden"
              style={{background:'#12121e', border:'1px solid #1e1e2e'}}>
              <div className="flex items-start justify-between mb-4">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{background: s.bg}}>
                  <Icon size={18} style={{color: s.color}} strokeWidth={2} />
                </div>
                <span className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full ${
                  s.trend === 'up' ? 'text-emerald-400 bg-emerald-400/10' : 'text-red-400 bg-red-400/10'
                }`}>
                  {s.trend === 'up'
                    ? <ArrowUpRight size={12} />
                    : <ArrowDownRight size={12} />
                  }
                  {s.trend === 'up' ? 'Good' : 'Watch'}
                </span>
              </div>
              <p className="text-[#4a4a6a] text-xs font-medium uppercase tracking-wider mb-1">{s.label}</p>
              <p className="text-2xl font-bold text-white mb-1" style={{color: s.color}}>{s.value}</p>
              <p className="text-[#4a4a6a] text-xs">{s.sub}</p>
            </div>
          )
        })}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-5 gap-4 mb-6">

        {/* Bar chart — spans 3 cols */}
        <div className="col-span-3 rounded-2xl p-6" style={{background:'#12121e', border:'1px solid #1e1e2e'}}>
          <div className="flex items-center justify-between mb-6">
            <div>
              <p className="text-white font-semibold text-sm">Monthly Overview</p>
              <p className="text-[#4a4a6a] text-xs mt-0.5">Income vs Spending</p>
            </div>
            <div className="flex items-center gap-4 text-xs text-[#4a4a6a]">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block bg-emerald-500"/>{' '}Income</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm inline-block bg-blue-500"/>{' '}Spent</span>
            </div>
          </div>
          {barData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} barGap={6} barCategoryGap="30%">
                <XAxis dataKey="month" tick={{fill:'#4a4a6a', fontSize:11}} axisLine={false} tickLine={false} />
                <YAxis tick={{fill:'#4a4a6a', fontSize:11}} axisLine={false} tickLine={false} tickFormatter={v => `$${v}`} />
                <Tooltip content={<CustomTooltip />} cursor={{fill:'rgba(255,255,255,0.03)'}} />
                <Bar dataKey="Income" fill="#10b981" radius={[6,6,0,0]} />
                <Bar dataKey="Spent"  fill="#3b82f6" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-52 flex flex-col items-center justify-center gap-2">
              <p className="text-[#4a4a6a] text-sm">No data yet</p>
              <p className="text-[#2a2a3a] text-xs">Upload a statement to see your monthly overview</p>
            </div>
          )}
        </div>

        {/* Pie chart — spans 2 cols */}
        <div className="col-span-2 rounded-2xl p-6" style={{background:'#12121e', border:'1px solid #1e1e2e'}}>
          <div className="mb-4">
            <p className="text-white font-semibold text-sm">Spending by Category</p>
            <p className="text-[#4a4a6a] text-xs mt-0.5">Where your money goes</p>
          </div>
          {pieData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={160}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%" cy="50%"
                    innerRadius={45} outerRadius={72}
                    dataKey="value"
                    paddingAngle={3}
                    strokeWidth={0}
                  >
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomPieTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-2 mt-2">
                {pieData.slice(0, 5).map((d, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{background: COLORS[i % COLORS.length]}} />
                      <span className="text-[#8888aa] text-xs truncate max-w-[100px]">{d.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1 rounded-full bg-[#1e1e2e] overflow-hidden">
                        <div className="h-full rounded-full" style={{
                          width: `${(d.value / totalSpent * 100).toFixed(0)}%`,
                          background: COLORS[i % COLORS.length]
                        }} />
                      </div>
                      <span className="text-white text-xs font-medium w-12 text-right">${d.value.toFixed(0)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="h-52 flex items-center justify-center">
              <p className="text-[#4a4a6a] text-sm">No expense data</p>
            </div>
          )}
        </div>

      </div>

      {/* Transactions table */}
      <div className="rounded-2xl overflow-hidden" style={{background:'#12121e', border:'1px solid #1e1e2e'}}>
        <div className="px-6 py-4 flex items-center justify-between" style={{borderBottom:'1px solid #1e1e2e'}}>
          <div>
            <p className="text-white text-sm font-semibold">Recent Transactions</p>
            <p className="text-[#4a4a6a] text-xs mt-0.5">{transactions.length} total records</p>
          </div>
        </div>
        {transactions.length === 0 ? (
          <div className="py-16 flex flex-col items-center gap-2">
            <p className="text-[#4a4a6a] text-sm">No transactions yet</p>
            <p className="text-[#2a2a3a] text-xs">Upload a bank statement to get started</p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr style={{borderBottom:'1px solid #1e1e2e'}}>
                {['Date','Description','Amount','Category'].map(h => (
                  <th key={h} className={`px-6 py-3 text-[#4a4a6a] text-xs font-medium uppercase tracking-wider ${h === 'Amount' ? 'text-right' : 'text-left'}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {transactions.slice(0, 20).map((t, i) => (
                <tr key={i} className="hover:bg-white/[0.02] transition-colors" style={{borderBottom:'1px solid #1a1a2a'}}>
                  <td className="px-6 py-3.5 text-[#6a6a8a] text-xs">{t.date}</td>
                  <td className="px-6 py-3.5 text-[#c0c0d8] font-medium">{t.description}</td>
                  <td className={`px-6 py-3.5 text-right font-semibold text-sm ${t.amount >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {t.amount >= 0 ? '+' : '-'}${Math.abs(t.amount).toFixed(2)}
                  </td>
                  <td className="px-6 py-3.5">
                    <span className="text-[#6a6a8a] text-xs px-2.5 py-1 rounded-lg" style={{background:'#1e1e2e'}}>
                      {t.category}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
