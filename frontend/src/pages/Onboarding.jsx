import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const CURRENCIES = ['USD','EUR','GBP','INR','AUD','CAD','SGD','AED','JPY','CHF']
const PAY_FREQUENCIES = ['Weekly','Bi-weekly','Monthly','Irregular']

const STEPS = [
  { id:'welcome',   title:'Welcome to FinanceAI',        subtitle:'Let\'s set up your financial profile — takes 2 minutes' },
  { id:'currency',  title:'What\'s your currency?',       subtitle:'Choose your base currency for all reports and budgets' },
  { id:'income',    title:'What\'s your monthly income?', subtitle:'Enter your take-home pay after tax. Helps us avoid double-counting bank deposits.' },
  { id:'budget',    title:'Set your monthly budget',      subtitle:'How much do you plan to spend each month?' },
  { id:'savings',   title:'What\'s your savings goal?',   subtitle:'How much would you like to save each month?' },
  { id:'frequency', title:'How often do you get paid?',   subtitle:'Helps us understand your cash flow pattern' },
  { id:'done',      title:'You\'re all set! 🎉',          subtitle:'Start by uploading a bank statement or entering transactions manually' },
]

export default function Onboarding() {
  const navigate = useNavigate()
  const [step, setStep] = useState(0)
  const [saving, setSaving] = useState(false)
  const [form, setForm] = useState({
    currency: 'USD',
    monthly_income: '',
    monthly_budget: '',
    savings_goal: '',
    pay_frequency: 'Monthly',
  })

  const current = STEPS[step]
  const progress = (step / (STEPS.length - 1)) * 100
  const next = () => setStep(s => Math.min(s + 1, STEPS.length - 1))
  const back = () => setStep(s => Math.max(s - 1, 0))

  const handleFinish = async (goTo) => {
    setSaving(true)
    try {
      await fetch('http://127.0.0.1:8000/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          monthly_income: parseFloat(form.monthly_income) || 0,
          monthly_budget: parseFloat(form.monthly_budget) || 0,
          savings_goal: parseFloat(form.savings_goal) || 0,
          currency: form.currency,
          pay_frequency: form.pay_frequency,
        })
      })
    } catch(e) {
      console.error(e)
    }
    localStorage.setItem('onboarding_complete', 'true')
    setSaving(false)
    navigate(goTo)
  }

  const S = {
    page: { minHeight:'100vh', background:'#0a0a0f', display:'flex', alignItems:'center', justifyContent:'center', fontFamily:'Inter,sans-serif', padding:24 },
    card: { background:'#12121e', border:'1px solid #1e1e2e', borderRadius:24, padding:'48px 52px', width:'100%', maxWidth:520 },
    logo: { display:'flex', alignItems:'center', gap:10, marginBottom:40 },
    logoBox: { width:36, height:36, borderRadius:10, background:'linear-gradient(135deg,#3b82f6,#1d4ed8)', display:'flex', alignItems:'center', justifyContent:'center', color:'#fff', fontWeight:800, fontSize:16 },
    progressBar: { background:'#1e1e2e', borderRadius:99, height:4, marginBottom:40, overflow:'hidden' },
    progressFill: { height:'100%', borderRadius:99, background:'linear-gradient(90deg,#3b82f6,#8b5cf6)', transition:'width 0.4s ease', width:`${progress}%` },
    stepNum: { color:'#4a4a6a', fontSize:12, fontWeight:600, letterSpacing:1, textTransform:'uppercase', marginBottom:12 },
    title: { color:'#fff', fontSize:26, fontWeight:800, marginBottom:8, letterSpacing:'-0.5px', lineHeight:1.2 },
    subtitle: { color:'#6a6a8a', fontSize:14, lineHeight:1.6, marginBottom:36 },
    label: { display:'block', color:'#8888aa', fontSize:11, fontWeight:600, letterSpacing:1.5, textTransform:'uppercase', marginBottom:10 },
    input: { width:'100%', background:'#0a0a0f', border:'1px solid #2a2a3a', borderRadius:12, padding:'14px 18px', color:'#fff', fontSize:16, outline:'none', boxSizing:'border-box', fontFamily:'Inter,sans-serif' },
    hint: { color:'#4a4a6a', fontSize:12, marginTop:10, lineHeight:1.6 },
    btnRow: { display:'flex', gap:12, marginTop:40 },
    btnPrimary: { flex:1, background:'#2563eb', color:'#fff', border:'none', borderRadius:12, padding:'14px', fontSize:15, fontWeight:700, cursor:'pointer', fontFamily:'Inter,sans-serif' },
    btnSecondary: { background:'#1e1e2e', color:'#8888aa', border:'1px solid #2a2a3a', borderRadius:12, padding:'14px 20px', fontSize:14, fontWeight:600, cursor:'pointer', fontFamily:'Inter,sans-serif' },
    option: (selected) => ({
      background: selected ? 'rgba(37,99,235,0.15)' : '#0a0a0f',
      border: `1px solid ${selected ? '#3b82f6' : '#2a2a3a'}`,
      borderRadius: 12, padding:'12px', color: selected ? '#fff' : '#8888aa',
      fontSize:14, fontWeight: selected ? 700 : 400, cursor:'pointer',
      textAlign:'center', transition:'all 0.15s', fontFamily:'Inter,sans-serif',
    }),
  }

  const currencySymbols = { USD:'$', EUR:'€', GBP:'£', INR:'₹', AUD:'A$', CAD:'C$', SGD:'S$', AED:'د.إ', JPY:'¥', CHF:'Fr' }
  const symbol = currencySymbols[form.currency] || '$'

  return (
    <div style={S.page}>
      <div style={S.card}>

        {/* Logo */}
        <div style={S.logo}>
          <div style={S.logoBox}>F</div>
          <span style={{color:'#fff', fontWeight:700, fontSize:16}}>FinanceAI</span>
        </div>

        {/* Progress bar */}
        <div style={S.progressBar}>
          <div style={S.progressFill}/>
        </div>

        {/* Step counter */}
        {step > 0 && step < STEPS.length - 1 && (
          <p style={S.stepNum}>Step {step} of {STEPS.length - 1}</p>
        )}

        <h2 style={S.title}>{current.title}</h2>
        <p style={S.subtitle}>{current.subtitle}</p>

        {/* ── WELCOME ── */}
        {step === 0 && (
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12}}>
            {[
              ['📊','Smart Dashboard','Auto-updating charts & metrics'],
              ['💬','AI Assistant','Ask anything about your spending'],
              ['📎','Any Bank','Upload from any bank worldwide'],
              ['🔒','Private','Data stays on your device only'],
            ].map(([icon,title,desc],i)=>(
              <div key={i} style={{background:'#0a0a0f', border:'1px solid #1e1e2e', borderRadius:12, padding:16}}>
                <div style={{fontSize:24, marginBottom:8}}>{icon}</div>
                <p style={{color:'#fff', fontWeight:600, fontSize:13, marginBottom:4}}>{title}</p>
                <p style={{color:'#4a4a6a', fontSize:12, lineHeight:1.5}}>{desc}</p>
              </div>
            ))}
          </div>
        )}

        {/* ── CURRENCY ── */}
        {step === 1 && (
          <div>
            <label style={S.label}>Select Your Currency</label>
            <div style={{display:'grid', gridTemplateColumns:'repeat(5,1fr)', gap:8}}>
              {CURRENCIES.map(c=>(
                <button key={c} onClick={()=>setForm({...form,currency:c})} style={S.option(form.currency===c)}>
                  <div style={{fontSize:14, marginBottom:2}}>{currencySymbols[c]}</div>
                  <div style={{fontSize:11}}>{c}</div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── INCOME ── */}
        {step === 2 && (
          <div>
            <label style={S.label}>Monthly Take-Home Income</label>
            <div style={{position:'relative'}}>
              <span style={{position:'absolute', left:16, top:'50%', transform:'translateY(-50%)', color:'#4a4a6a', fontSize:18}}>{symbol}</span>
              <input
                type="number"
                style={{...S.input, paddingLeft:40}}
                placeholder="e.g. 4500"
                value={form.monthly_income}
                onChange={e=>setForm({...form,monthly_income:e.target.value})}
                autoFocus
              />
            </div>
            <p style={S.hint}>
              💡 Enter your <strong style={{color:'#8888aa'}}>take-home pay after tax</strong>. If your bank statement shows direct deposits, we'll use this number to avoid counting your income twice in reports.
            </p>
          </div>
        )}

        {/* ── BUDGET ── */}
        {step === 3 && (
          <div>
            <label style={S.label}>Total Monthly Budget</label>
            <div style={{position:'relative'}}>
              <span style={{position:'absolute', left:16, top:'50%', transform:'translateY(-50%)', color:'#4a4a6a', fontSize:18}}>{symbol}</span>
              <input
                type="number"
                style={{...S.input, paddingLeft:40}}
                placeholder="e.g. 3000"
                value={form.monthly_budget}
                onChange={e=>setForm({...form,monthly_budget:e.target.value})}
                autoFocus
              />
            </div>
            {form.monthly_income && (
              <p style={S.hint}>
                💡 Based on your income of <strong style={{color:'#fff'}}>{symbol}{parseFloat(form.monthly_income).toLocaleString()}</strong>, a good rule is to budget <strong style={{color:'#8888aa'}}>80%</strong> ({symbol}{(parseFloat(form.monthly_income)*0.8).toFixed(0)}) and save the rest.
              </p>
            )}
          </div>
        )}

        {/* ── SAVINGS GOAL ── */}
        {step === 4 && (
          <div>
            <label style={S.label}>Monthly Savings Goal</label>
            <div style={{position:'relative'}}>
              <span style={{position:'absolute', left:16, top:'50%', transform:'translateY(-50%)', color:'#4a4a6a', fontSize:18}}>{symbol}</span>
              <input
                type="number"
                style={{...S.input, paddingLeft:40}}
                placeholder="e.g. 500"
                value={form.savings_goal}
                onChange={e=>setForm({...form,savings_goal:e.target.value})}
                autoFocus
              />
            </div>
            {form.monthly_income && form.monthly_budget && (
              <div style={{marginTop:12, background:'rgba(16,185,129,0.08)', border:'1px solid rgba(16,185,129,0.2)', borderRadius:10, padding:'12px 16px'}}>
                <p style={{color:'#10b981', fontSize:13, fontWeight:600}}>
                  You could save up to {symbol}{Math.max(0, parseFloat(form.monthly_income||0) - parseFloat(form.monthly_budget||0)).toFixed(0)}/month
                </p>
                <p style={{color:'#4a4a6a', fontSize:12, marginTop:4}}>Based on your income minus budget</p>
              </div>
            )}
          </div>
        )}

        {/* ── PAY FREQUENCY ── */}
        {step === 5 && (
          <div>
            <label style={S.label}>How Often Do You Get Paid?</label>
            <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12}}>
              {[
                ['Weekly','📅','Every 7 days'],
                ['Bi-weekly','📆','Every 2 weeks'],
                ['Monthly','🗓️','Once a month'],
                ['Irregular','🔄','Varies each month'],
              ].map(([freq,icon,desc])=>(
                <button key={freq} onClick={()=>setForm({...form,pay_frequency:freq})}
                  style={{...S.option(form.pay_frequency===freq), padding:18, textAlign:'left'}}>
                  <div style={{fontSize:24, marginBottom:6}}>{icon}</div>
                  <p style={{fontSize:14, fontWeight:600, marginBottom:2}}>{freq}</p>
                  <p style={{fontSize:11, color: form.pay_frequency===freq?'rgba(255,255,255,0.6)':'#4a4a6a'}}>{desc}</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── DONE ── */}
        {step === STEPS.length - 1 && (
          <div>
            {/* Summary */}
            <div style={{background:'#0a0a0f', border:'1px solid #1e1e2e', borderRadius:14, padding:20, marginBottom:28}}>
              {[
                ['Currency',       form.currency],
                ['Monthly Income', `${symbol}${parseFloat(form.monthly_income||0).toLocaleString()}`],
                ['Monthly Budget', `${symbol}${parseFloat(form.monthly_budget||0).toLocaleString()}`],
                ['Savings Goal',   `${symbol}${parseFloat(form.savings_goal||0).toLocaleString()}`],
                ['Pay Frequency',  form.pay_frequency],
              ].map(([label,value],i,arr)=>(
                <div key={i} style={{display:'flex', justifyContent:'space-between', padding:'10px 0', borderBottom:i<arr.length-1?'1px solid #1e1e2e':'none'}}>
                  <span style={{color:'#6a6a8a', fontSize:13}}>{label}</span>
                  <span style={{color:'#fff', fontWeight:700, fontSize:13}}>{value}</span>
                </div>
              ))}
            </div>

            <p style={{color:'#8888aa', fontSize:13, textAlign:'center', marginBottom:16, fontWeight:500}}>How would you like to add your transactions?</p>
            <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:12}}>
              <button onClick={()=>handleFinish('/app/upload')} disabled={saving}
                style={{background:'#2563eb', color:'#fff', border:'none', borderRadius:16, padding:'20px 16px', cursor:'pointer', fontFamily:'Inter,sans-serif', opacity:saving?0.6:1}}>
                <div style={{fontSize:32, marginBottom:10}}>📎</div>
                <p style={{fontWeight:700, fontSize:15, marginBottom:4}}>Upload Statement</p>
                <p style={{color:'rgba(255,255,255,0.6)', fontSize:12}}>Import from any bank</p>
              </button>
              <button onClick={()=>handleFinish('/app/upload')} disabled={saving}
                style={{background:'#1e1e2e', color:'#fff', border:'1px solid #2a2a3a', borderRadius:16, padding:'20px 16px', cursor:'pointer', fontFamily:'Inter,sans-serif', opacity:saving?0.6:1}}>
                <div style={{fontSize:32, marginBottom:10}}>✏️</div>
                <p style={{fontWeight:700, fontSize:15, marginBottom:4}}>Enter Manually</p>
                <p style={{color:'#6a6a8a', fontSize:12}}>Type in transactions</p>
              </button>
            </div>
            <p style={{color:'#4a4a6a', fontSize:12, textAlign:'center', marginTop:16}}>You can update these settings anytime</p>
          </div>
        )}

        {/* ── NAVIGATION BUTTONS ── */}
        {step < STEPS.length - 1 && (
          <div style={S.btnRow}>
            {step > 0 && (
              <button onClick={back} style={S.btnSecondary}>← Back</button>
            )}
            <button onClick={next} style={S.btnPrimary}>
              {step === 0 ? 'Get Started →' : step === STEPS.length - 2 ? 'Review →' : 'Continue →'}
            </button>
          </div>
        )}

        {/* Skip */}
        {step > 0 && step < STEPS.length - 1 && (
          <p onClick={next} style={{textAlign:'center', color:'#4a4a6a', fontSize:12, marginTop:16, cursor:'pointer'}}>
            Skip this step
          </p>
        )}

      </div>
    </div>
  )
}
