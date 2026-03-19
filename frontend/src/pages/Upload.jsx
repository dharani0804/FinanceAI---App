import { useState } from 'react'

const CATEGORIES = [
  'Uncategorized','Food & Dining','Shopping','Transport',
  'Entertainment','Bills & Utilities','Health','Travel',
  'Groceries','Salary','Transfer','Other'
]

const CURRENCIES = ['USD','EUR','GBP','INR','AUD','CAD','SGD','AED','JPY','CHF']

const emptyLine = {
  date: new Date().toISOString().split('T')[0],
  description: '',
  amount: '',
  currency: 'USD',
  category: 'Uncategorized',
}

export default function Upload() {
  const [tab, setTab] = useState('upload')

  // Upload state
  const [file, setFile] = useState(null)
  const [bankName, setBankName] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [uploadedTx, setUploadedTx] = useState([])
  const [drag, setDrag] = useState(false)

  // Manual state — multiple lines
  const [lines, setLines] = useState([{ ...emptyLine, id: Date.now() }])
  const [saving, setSaving] = useState(false)
  const [manualStatus, setManualStatus] = useState('')
  const [savedTx, setSavedTx] = useState([])
  const [manualBank, setManualBank] = useState('')

  // Edit state
  const [editingId, setEditingId] = useState(null)
  const [editRow, setEditRow] = useState({})

  const addLine = () => setLines(prev => [...prev, { ...emptyLine, id: Date.now() }])

  const updateLine = (id, field, value) => {
    setLines(prev => prev.map(l => l.id === id ? { ...l, [field]: value } : l))
  }

  const removeLine = (id) => {
    if (lines.length === 1) return
    setLines(prev => prev.filter(l => l.id !== id))
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setUploadStatus('')
    const formData = new FormData()
    formData.append('file', file)
    const bank = bankName.trim() || 'Unknown Bank'
    try {
      const res = await fetch(`http://127.0.0.1:8000/upload?bank_name=${encodeURIComponent(bank)}`, {
        method: 'POST', body: formData
      })
      const data = await res.json()
      if (data.success) {
        setUploadStatus(`✓ ${data.transactions_imported} transactions imported from ${bank}`)
        setUploadedTx(data.transactions)
        setFile(null)
        setBankName('')
      } else {
        setUploadStatus(`✗ ${data.error}`)
      }
    } catch {
      setUploadStatus('✗ Could not connect to backend')
    } finally {
      setUploading(false)
    }
  }

  const handleSaveAll = async () => {
    const valid = lines.filter(l => l.description.trim() && l.amount && l.date)
    if (valid.length === 0) {
      setManualStatus('✗ Fill in at least one complete row')
      return
    }
    setSaving(true)
    setManualStatus('')
    const results = []
    for (const line of valid) {
      try {
        const res = await fetch('http://127.0.0.1:8000/transactions/manual', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            date: line.date,
            description: line.description,
            amount: parseFloat(line.amount),
            currency: line.currency,
            category: line.category,
            bank_source: manualBank.trim() || 'Manual Entry',
          })
        })
        const data = await res.json()
        if (data.success) results.push(data.transaction)
      } catch {}
    }
    setSaving(false)
    if (results.length > 0) {
      setManualStatus(`✓ ${results.length} transaction${results.length > 1 ? 's' : ''} saved!`)
      setSavedTx(prev => [...results, ...prev])
      setLines([{ ...emptyLine, id: Date.now() }])
      setManualBank('')
    } else {
      setManualStatus('✗ Failed to save. Check your backend.')
    }
  }

  const startEdit = (tx) => { setEditingId(tx.id); setEditRow({ ...tx }) }
  const cancelEdit = () => setEditingId(null)

  const saveEdit = async (id) => {
    try {
      await fetch(`http://127.0.0.1:8000/transactions/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editRow)
      })
      setSavedTx(prev => prev.map(t => t.id === id ? { ...t, ...editRow } : t))
      setUploadedTx(prev => prev.map(t => t.id === id ? { ...t, ...editRow } : t))
    } catch {}
    setEditingId(null)
  }

  const S = {
    page: { padding: '40px 56px', maxWidth: 1100, margin: '0 auto', fontFamily: 'Inter, sans-serif' },
    title: { fontSize: 24, fontWeight: 800, color: '#fff', marginBottom: 6, letterSpacing: '-0.5px' },
    subtitle: { color: '#6a6a8a', fontSize: 14, marginBottom: 32 },
    tabs: { display: 'flex', gap: 4, background: '#12121e', border: '1px solid #1e1e2e', borderRadius: 14, padding: 4, marginBottom: 28, width: 'fit-content' },
    tab: (a) => ({ padding: '10px 28px', borderRadius: 10, fontSize: 14, fontWeight: 600, cursor: 'pointer', border: 'none', background: a ? '#2563eb' : 'transparent', color: a ? '#fff' : '#6a6a8a', boxShadow: a ? '0 2px 12px rgba(37,99,235,0.3)' : 'none' }),
    card: { background: '#12121e', border: '1px solid #1e1e2e', borderRadius: 20, padding: 32 },
    label: { display: 'block', color: '#8888aa', fontSize: 11, fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 7 },
    input: { width: '100%', background: '#0a0a0f', border: '1px solid #2a2a3a', borderRadius: 10, padding: '10px 14px', color: '#fff', fontSize: 14, outline: 'none', boxSizing: 'border-box' },
    select: { width: '100%', background: '#0a0a0f', border: '1px solid #2a2a3a', borderRadius: 10, padding: '10px 14px', color: '#fff', fontSize: 14, outline: 'none', boxSizing: 'border-box' },
    dropzone: (d) => ({ background: d ? 'rgba(37,99,235,0.06)' : '#0a0a0f', border: `2px dashed ${d ? '#3b82f6' : '#2a2a3a'}`, borderRadius: 16, padding: '52px 32px', textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s', marginBottom: 20 }),
    btn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 12, padding: '12px 28px', fontSize: 14, fontWeight: 700, cursor: 'pointer' },
    btnSm: { background: '#1e1e2e', color: '#8888aa', border: '1px solid #2a2a3a', borderRadius: 8, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer' },
    btnDanger: { background: 'none', color: '#4a4a6a', border: '1px solid #2a2a3a', borderRadius: 8, padding: '6px 10px', fontSize: 13, cursor: 'pointer' },
    status: (ok) => ({ marginTop: 14, fontSize: 13, fontWeight: 600, color: ok ? '#10b981' : '#ef4444' }),
    th: { padding: '11px 14px', background: '#0a0a0f', color: '#4a4a6a', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, textAlign: 'left', borderBottom: '1px solid #1e1e2e' },
    td: { padding: '11px 14px', color: '#c0c0d8', fontSize: 13, borderBottom: '1px solid #1a1a2a', verticalAlign: 'middle' },
    editInput: { background: '#0a0a0f', border: '1px solid #3b82f6', borderRadius: 6, padding: '5px 8px', color: '#fff', fontSize: 12, width: '100%', outline: 'none' },
    editSelect: { background: '#0a0a0f', border: '1px solid #3b82f6', borderRadius: 6, padding: '5px 8px', color: '#fff', fontSize: 12, width: '100%', outline: 'none' },
  }

  const renderRow = (tx) => {
    const isEditing = editingId === tx.id
    return (
      <tr key={tx.id}>
        <td style={S.td}>{isEditing ? <input style={S.editInput} value={editRow.date||''} onChange={e=>setEditRow({...editRow,date:e.target.value})}/> : <span style={{color:'#6a6a8a'}}>{tx.date}</span>}</td>
        <td style={S.td}>{isEditing ? <input style={S.editInput} value={editRow.description||''} onChange={e=>setEditRow({...editRow,description:e.target.value})}/> : tx.description}</td>
        <td style={{...S.td,textAlign:'right'}}>{isEditing ? <input style={{...S.editInput,width:80}} value={editRow.amount||''} onChange={e=>setEditRow({...editRow,amount:e.target.value})}/> : <span style={{color:parseFloat(tx.amount)>=0?'#10b981':'#ef4444',fontWeight:600}}>{parseFloat(tx.amount)>=0?'+':'-'}${Math.abs(parseFloat(tx.amount)).toFixed(2)}</span>}</td>
        <td style={S.td}>{isEditing ? <select style={S.editSelect} value={editRow.currency||'USD'} onChange={e=>setEditRow({...editRow,currency:e.target.value})}>{CURRENCIES.map(c=><option key={c}>{c}</option>)}</select> : tx.currency}</td>
        <td style={S.td}>{isEditing ? <select style={S.editSelect} value={editRow.category||'Uncategorized'} onChange={e=>setEditRow({...editRow,category:e.target.value})}>{CATEGORIES.map(c=><option key={c}>{c}</option>)}</select> : <span style={{background:'#1e1e2e',color:'#8888aa',fontSize:11,padding:'3px 10px',borderRadius:6}}>{tx.category}</span>}</td>
        <td style={S.td}>{isEditing ? <input style={S.editInput} value={editRow.bank_source||''} onChange={e=>setEditRow({...editRow,bank_source:e.target.value})}/> : <span style={{color:'#4a4a6a',fontSize:12}}>{tx.bank_source}</span>}</td>
        <td style={S.td}>
          {isEditing
            ? <div style={{display:'flex',gap:6}}>
                <button onClick={()=>saveEdit(tx.id)} style={{...S.btnSm,background:'#10b981',color:'#fff',border:'none'}}>Save</button>
                <button onClick={cancelEdit} style={S.btnSm}>✕</button>
              </div>
            : <button onClick={()=>startEdit(tx)} style={S.btnSm}>Edit</button>
          }
        </td>
      </tr>
    )
  }

  return (
    <div style={S.page}>
      <h1 style={S.title}>Add Transactions</h1>
      <p style={S.subtitle}>Upload a bank statement or enter transactions line by line — all fields editable after saving</p>

      {/* TABS */}
      <div style={S.tabs}>
        <button style={S.tab(tab==='upload')} onClick={()=>setTab('upload')}>📎 Upload Statement</button>
        <button style={S.tab(tab==='manual')} onClick={()=>setTab('manual')}>✏️ Enter Manually</button>
      </div>

      {/* ── UPLOAD TAB ── */}
      {tab === 'upload' && (
        <div style={S.card}>
          <p style={{color:'#8888aa',fontSize:13,marginBottom:24}}>Supports bank statements, CSV files and transaction documents. Your file is never stored — only transaction data is saved.</p>

          {/* Dropzone */}
          <div
            style={S.dropzone(drag)}
            onDragOver={e=>{e.preventDefault();setDrag(true)}}
            onDragLeave={()=>setDrag(false)}
            onDrop={e=>{e.preventDefault();setDrag(false);setFile(e.dataTransfer.files[0])}}
            onClick={()=>document.getElementById('fileInput').click()}
          >
            <div style={{fontSize:40,marginBottom:12}}>📎</div>
            <p style={{color:file?'#fff':'#8888aa',fontWeight:700,fontSize:16,marginBottom:6}}>
              {file ? file.name : 'Drop your file here'}
            </p>
            <p style={{color:'#4a4a6a',fontSize:13}}>
              {file ? `${(file.size/1024).toFixed(1)} KB · Ready to import` : 'CSV · PDF · XLS · click to browse'}
            </p>
            <input id="fileInput" type="file" accept=".pdf,.csv,.xlsx,.xls,.doc,.docx" style={{display:'none'}} onChange={e=>setFile(e.target.files[0])}/>
          </div>

          {/* Bank name */}
          <div style={{marginBottom:20}}>
            <label style={S.label}>Bank / Account Name <span style={{color:'#4a4a6a',textTransform:'none',letterSpacing:0}}>(optional)</span></label>
            <input style={S.input} placeholder="e.g. Chase, HDFC, Barclays, Revolut..." value={bankName} onChange={e=>setBankName(e.target.value)}/>
          </div>

          <button onClick={handleUpload} disabled={!file||uploading} style={{...S.btn,opacity:(!file||uploading)?0.4:1,width:'100%'}}>
            {uploading ? '⏳ Parsing statement...' : '↑ Upload & Import'}
          </button>

          {uploadStatus && <p style={S.status(uploadStatus.startsWith('✓'))}>{uploadStatus}</p>}

          {/* Uploaded results table */}
          {uploadedTx.length > 0 && (
            <div style={{marginTop:28,borderRadius:16,overflow:'hidden',border:'1px solid #1e1e2e'}}>
              <div style={{padding:'14px 16px',borderBottom:'1px solid #1e1e2e',display:'flex',justifyContent:'space-between',alignItems:'center',background:'#0a0a0f'}}>
                <p style={{color:'#fff',fontWeight:700,fontSize:14}}>Imported — {uploadedTx.length} transactions</p>
                <span style={{color:'#4a4a6a',fontSize:12}}>Click Edit to modify any field</span>
              </div>
              <div style={{overflowX:'auto'}}>
                <table style={{width:'100%',borderCollapse:'collapse'}}>
                  <thead><tr>{['Date','Description','Amount','Currency','Category','Bank',''].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                  <tbody>{uploadedTx.map(tx=>renderRow(tx))}</tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── MANUAL TAB ── */}
      {tab === 'manual' && (
        <div style={S.card}>
          <p style={{color:'#8888aa',fontSize:13,marginBottom:24}}>Add transactions one line at a time. Click <strong style={{color:'#fff'}}>+ Add Row</strong> for more entries. Save all at once when ready.</p>

          {/* Bank name for manual */}
          <div style={{marginBottom:20,maxWidth:320}}>
            <label style={S.label}>Bank / Account Name <span style={{color:'#4a4a6a',textTransform:'none',letterSpacing:0}}>(optional)</span></label>
            <input style={S.input} placeholder="e.g. Chase, HDFC, Revolut..." value={manualBank} onChange={e=>setManualBank(e.target.value)}/>
          </div>

          {/* Line by line entry table */}
          <div style={{borderRadius:14,overflow:'hidden',border:'1px solid #1e1e2e',marginBottom:16}}>
            {/* Header */}
            <div style={{display:'grid',gridTemplateColumns:'130px 1fr 120px 90px 170px 36px',gap:0,background:'#0a0a0f',borderBottom:'1px solid #1e1e2e'}}>
              {['Date','Description / Merchant','Amount','Currency','Category',''].map((h,i)=>(
                <div key={i} style={{padding:'10px 14px',color:'#4a4a6a',fontSize:11,fontWeight:600,textTransform:'uppercase',letterSpacing:1}}>{h}</div>
              ))}
            </div>

            {/* Lines */}
            {lines.map((line, idx) => (
              <div key={line.id} style={{display:'grid',gridTemplateColumns:'130px 1fr 120px 90px 170px 36px',gap:0,borderBottom:'1px solid #1a1a2a',background: idx%2===0?'#12121e':'#0f0f1a',alignItems:'center'}}>
                <div style={{padding:'8px 10px'}}>
                  <input
                    type="date"
                    style={{...S.input,padding:'7px 10px',fontSize:12}}
                    value={line.date}
                    onChange={e=>updateLine(line.id,'date',e.target.value)}
                  />
                </div>
                <div style={{padding:'8px 10px'}}>
                  <input
                    style={{...S.input,padding:'7px 10px',fontSize:13}}
                    placeholder="e.g. Starbucks, Netflix, Salary..."
                    value={line.description}
                    onChange={e=>updateLine(line.id,'description',e.target.value)}
                  />
                </div>
                <div style={{padding:'8px 10px'}}>
                  <input
                    type="number"
                    style={{...S.input,padding:'7px 10px',fontSize:13}}
                    placeholder="-45.99"
                    value={line.amount}
                    onChange={e=>updateLine(line.id,'amount',e.target.value)}
                  />
                </div>
                <div style={{padding:'8px 10px'}}>
                  <select
                    style={{...S.select,padding:'7px 10px',fontSize:12}}
                    value={line.currency}
                    onChange={e=>updateLine(line.id,'currency',e.target.value)}
                  >
                    {CURRENCIES.map(c=><option key={c}>{c}</option>)}
                  </select>
                </div>
                <div style={{padding:'8px 10px'}}>
                  <select
                    style={{...S.select,padding:'7px 8px',fontSize:12}}
                    value={line.category}
                    onChange={e=>updateLine(line.id,'category',e.target.value)}
                  >
                    {CATEGORIES.map(c=><option key={c}>{c}</option>)}
                  </select>
                </div>
                <div style={{padding:'8px 4px',textAlign:'center'}}>
                  <button onClick={()=>removeLine(line.id)} style={{background:'none',border:'none',color:'#4a4a6a',fontSize:16,cursor:'pointer',lineHeight:1}} title="Remove row">✕</button>
                </div>
              </div>
            ))}
          </div>

          {/* Add row + hint */}
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:24}}>
            <button onClick={addLine} style={{...S.btnSm,display:'flex',alignItems:'center',gap:6,fontSize:13}}>
              + Add Row
            </button>
            <p style={{color:'#4a4a6a',fontSize:12}}>Use − for expenses (−45.99) and + for income (3500)</p>
          </div>

          {/* Save button */}
          <button onClick={handleSaveAll} disabled={saving} style={{...S.btn,opacity:saving?0.5:1,minWidth:180}}>
            {saving ? 'Saving...' : `💾 Save ${lines.filter(l=>l.description&&l.amount).length} Transaction${lines.filter(l=>l.description&&l.amount).length!==1?'s':''}`}
          </button>

          {manualStatus && <p style={S.status(manualStatus.startsWith('✓'))}>{manualStatus}</p>}

          {/* Saved transactions */}
          {savedTx.length > 0 && (
            <div style={{marginTop:28,borderRadius:16,overflow:'hidden',border:'1px solid #1e1e2e'}}>
              <div style={{padding:'14px 16px',borderBottom:'1px solid #1e1e2e',display:'flex',justifyContent:'space-between',alignItems:'center',background:'#0a0a0f'}}>
                <p style={{color:'#fff',fontWeight:700,fontSize:14}}>Saved — {savedTx.length} transactions</p>
                <span style={{color:'#4a4a6a',fontSize:12}}>Click Edit to modify any field</span>
              </div>
              <div style={{overflowX:'auto'}}>
                <table style={{width:'100%',borderCollapse:'collapse'}}>
                  <thead><tr>{['Date','Description','Amount','Currency','Category','Bank',''].map(h=><th key={h} style={S.th}>{h}</th>)}</tr></thead>
                  <tbody>{savedTx.map(tx=>renderRow(tx))}</tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
