import { useState } from 'react'

const CATEGORIES = [
  'Uncategorized','Food & Dining','Shopping','Transport',
  'Entertainment','Bills & Utilities','Health','Travel',
  'Groceries','Salary','Transfer','Other'
]

const CURRENCIES = ['USD','EUR','GBP','INR','AUD','CAD','SGD','AED','JPY','CHF']

const emptyForm = {
  date: new Date().toISOString().split('T')[0],
  description: '',
  amount: '',
  currency: 'USD',
  category: 'Uncategorized',
  bank_source: 'Manual Entry',
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

  // Manual state
  const [form, setForm] = useState(emptyForm)
  const [saving, setSaving] = useState(false)
  const [manualStatus, setManualStatus] = useState('')
  const [recentManual, setRecentManual] = useState([])

  // Edit state
  const [editingId, setEditingId] = useState(null)
  const [editRow, setEditRow] = useState({})

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

  const handleManualSave = async () => {
    if (!form.description || !form.amount || !form.date) {
      setManualStatus('✗ Please fill in date, description and amount')
      return
    }
    setSaving(true)
    setManualStatus('')
    try {
      const res = await fetch('http://127.0.0.1:8000/transactions/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, amount: parseFloat(form.amount) })
      })
      const data = await res.json()
      if (data.success) {
        setManualStatus('✓ Transaction saved!')
        setRecentManual(prev => [data.transaction, ...prev])
        setForm(emptyForm)
      } else {
        setManualStatus(`✗ ${data.error}`)
      }
    } catch {
      setManualStatus('✗ Could not connect to backend')
    } finally {
      setSaving(false)
    }
  }

  const startEdit = (tx) => {
    setEditingId(tx.id || tx.description)
    setEditRow({ ...tx })
  }

  const saveEdit = async (tx) => {
    try {
      await fetch(`http://127.0.0.1:8000/transactions/${tx.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editRow)
      })
      setRecentManual(prev => prev.map(t => t.id === tx.id ? { ...t, ...editRow } : t))
      setUploadedTx(prev => prev.map(t => t.id === tx.id ? { ...t, ...editRow } : t))
    } catch {}
    setEditingId(null)
  }

  const S = {
    page: { padding: '40px 64px', maxWidth: 1000, margin: '0 auto', fontFamily: 'Inter, sans-serif' },
    header: { marginBottom: 32 },
    title: { fontSize: 26, fontWeight: 800, color: '#fff', marginBottom: 6, letterSpacing: '-0.5px' },
    subtitle: { color: '#6a6a8a', fontSize: 14 },
    tabs: { display: 'flex', gap: 4, background: '#12121e', border: '1px solid #1e1e2e', borderRadius: 14, padding: 4, marginBottom: 32, width: 'fit-content' },
    tab: (active) => ({ padding: '10px 28px', borderRadius: 10, fontSize: 14, fontWeight: 600, cursor: 'pointer', border: 'none', transition: 'all 0.2s', background: active ? '#2563eb' : 'transparent', color: active ? '#fff' : '#6a6a8a', boxShadow: active ? '0 2px 12px rgba(37,99,235,0.3)' : 'none' }),
    card: { background: '#12121e', border: '1px solid #1e1e2e', borderRadius: 20, padding: 32 },
    label: { display: 'block', color: '#8888aa', fontSize: 12, fontWeight: 600, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 8 },
    input: { width: '100%', background: '#0a0a0f', border: '1px solid #2a2a3a', borderRadius: 10, padding: '10px 14px', color: '#fff', fontSize: 14, outline: 'none', boxSizing: 'border-box' },
    select: { width: '100%', background: '#0a0a0f', border: '1px solid #2a2a3a', borderRadius: 10, padding: '10px 14px', color: '#fff', fontSize: 14, outline: 'none', boxSizing: 'border-box' },
    grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 },
    grid3: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 16 },
    saveBtn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 12, padding: '12px 32px', fontSize: 14, fontWeight: 700, cursor: 'pointer', marginTop: 8 },
    status: (ok) => ({ marginTop: 12, fontSize: 13, fontWeight: 600, color: ok ? '#10b981' : '#ef4444' }),
    dropzone: (drag) => ({ background: drag ? 'rgba(37,99,235,0.08)' : '#0a0a0f', border: `2px dashed ${drag ? '#3b82f6' : '#2a2a3a'}`, borderRadius: 16, padding: '48px 32px', textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s', marginBottom: 20 }),
    tableWrap: { marginTop: 28, borderRadius: 16, overflow: 'hidden', border: '1px solid #1e1e2e' },
    th: { padding: '12px 16px', background: '#0a0a0f', color: '#4a4a6a', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1, textAlign: 'left', borderBottom: '1px solid #1e1e2e' },
    td: { padding: '12px 16px', color: '#c0c0d8', fontSize: 13, borderBottom: '1px solid #1a1a2a' },
    editInput: { background: '#0a0a0f', border: '1px solid #3b82f6', borderRadius: 6, padding: '4px 8px', color: '#fff', fontSize: 12, width: '100%' },
    editSelect: { background: '#0a0a0f', border: '1px solid #3b82f6', borderRadius: 6, padding: '4px 8px', color: '#fff', fontSize: 12, width: '100%' },
  }

  const renderEditableRow = (tx, onSave) => {
    const isEditing = editingId === (tx.id || tx.description)
    return (
      <tr key={tx.id || tx.description} style={{borderBottom:'1px solid #1a1a2a'}}>
        <td style={S.td}>
          {isEditing
            ? <input style={S.editInput} value={editRow.date || ''} onChange={e=>setEditRow({...editRow,date:e.target.value})} />
            : <span style={{color:'#6a6a8a'}}>{tx.date}</span>}
        </td>
        <td style={S.td}>
          {isEditing
            ? <input style={S.editInput} value={editRow.description||''} onChange={e=>setEditRow({...editRow,description:e.target.value})} />
            : tx.description}
        </td>
        <td style={{...S.td, textAlign:'right'}}>
          {isEditing
            ? <input style={{...S.editInput, width:80}} value={editRow.amount||''} onChange={e=>setEditRow({...editRow,amount:e.target.value})} />
            : <span style={{color: parseFloat(tx.amount)>=0?'#10b981':'#ef4444', fontWeight:600}}>
                {parseFloat(tx.amount)>=0?'+':''}{Math.abs(parseFloat(tx.amount)).toFixed(2)}
              </span>}
        </td>
        <td style={S.td}>
          {isEditing
            ? <select style={S.editSelect} value={editRow.currency||'USD'} onChange={e=>setEditRow({...editRow,currency:e.target.value})}>
                {CURRENCIES.map(c=><option key={c}>{c}</option>)}
              </select>
            : tx.currency}
        </td>
        <td style={S.td}>
          {isEditing
            ? <select style={S.editSelect} value={editRow.category||'Uncategorized'} onChange={e=>setEditRow({...editRow,category:e.target.value})}>
                {CATEGORIES.map(c=><option key={c}>{c}</option>)}
              </select>
            : <span style={{background:'#1e1e2e', color:'#8888aa', fontSize:11, padding:'3px 10px', borderRadius:6}}>{tx.category}</span>}
        </td>
        <td style={S.td}>
          {isEditing
            ? <input style={S.editInput} value={editRow.bank_source||''} onChange={e=>setEditRow({...editRow,bank_source:e.target.value})} />
            : <span style={{color:'#4a4a6a', fontSize:12}}>{tx.bank_source}</span>}
        </td>
        <td style={S.td}>
          {isEditing ? (
            <div style={{display:'flex', gap:8}}>
              <button onClick={()=>saveEdit(tx)} style={{background:'#10b981', color:'#fff', border:'none', borderRadius:6, padding:'4px 12px', fontSize:12, cursor:'pointer', fontWeight:600}}>Save</button>
              <button onClick={()=>setEditingId(null)} style={{background:'#2a2a3a', color:'#fff', border:'none', borderRadius:6, padding:'4px 10px', fontSize:12, cursor:'pointer'}}>✕</button>
            </div>
          ) : (
            <button onClick={()=>startEdit(tx)} style={{background:'none', color:'#4a4a6a', border:'1px solid #2a2a3a', borderRadius:6, padding:'4px 12px', fontSize:12, cursor:'pointer'}}>Edit</button>
          )}
        </td>
      </tr>
    )
  }

  return (
    <div style={S.page}>
      <div style={S.header}>
        <h1 style={S.title}>Add Transactions</h1>
        <p style={S.subtitle}>Upload a bank statement or enter transactions manually — every field is editable after saving</p>
      </div>

      {/* TABS */}
      <div style={S.tabs}>
        <button style={S.tab(tab==='upload')} onClick={()=>setTab('upload')}>📎 Upload Statement</button>
        <button style={S.tab(tab==='manual')} onClick={()=>setTab('manual')}>✏️ Enter Manually</button>
      </div>

      {/* UPLOAD TAB */}
      {tab === 'upload' && (
        <div style={S.card}>
          <div
            style={S.dropzone(drag)}
            onDragOver={e=>{e.preventDefault();setDrag(true)}}
            onDragLeave={()=>setDrag(false)}
            onDrop={e=>{e.preventDefault();setDrag(false);setFile(e.dataTransfer.files[0])}}
            onClick={()=>document.getElementById('fileInput').click()}
          >
            <div style={{fontSize:36, marginBottom:12}}>📎</div>
            <p style={{color:file?'#fff':'#6a6a8a', fontWeight:600, fontSize:15, marginBottom:6}}>
              {file ? file.name : 'Drop your statement here'}
            </p>
            <p style={{color:'#4a4a6a', fontSize:13}}>
              {file ? `${(file.size/1024).toFixed(1)} KB · Ready to upload` : 'or click to browse · CSV or PDF'}
            </p>
            <input id="fileInput" type="file" accept=".pdf,.csv" style={{display:'none'}} onChange={e=>setFile(e.target.files[0])} />
          </div>

          <div style={{marginBottom:20}}>
            <label style={S.label}>Bank / Account Name</label>
            <input
              style={S.input}
              placeholder="e.g. Chase, HDFC, Barclays..."
              value={bankName}
              onChange={e=>setBankName(e.target.value)}
            />
          </div>

          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            style={{...S.saveBtn, opacity: (!file||uploading)?0.5:1, width:'100%'}}
          >
            {uploading ? '⏳ Parsing...' : 'Upload & Import'}
          </button>

          {uploadStatus && (
            <p style={S.status(uploadStatus.startsWith('✓'))}>{uploadStatus}</p>
          )}

          {uploadedTx.length > 0 && (
            <div style={S.tableWrap}>
              <div style={{padding:'14px 16px', borderBottom:'1px solid #1e1e2e', display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                <p style={{color:'#fff', fontWeight:600, fontSize:14}}>Imported Transactions</p>
                <span style={{color:'#4a4a6a', fontSize:12}}>{uploadedTx.length} records · click Edit to modify any field</span>
              </div>
              <table style={{width:'100%', borderCollapse:'collapse'}}>
                <thead>
                  <tr>
                    {['Date','Description','Amount','Currency','Category','Bank','Actions'].map(h=>(
                      <th key={h} style={S.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadedTx.map(tx => renderEditableRow(tx, saveEdit))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* MANUAL TAB */}
      {tab === 'manual' && (
        <div style={S.card}>
          <p style={{color:'#8888aa', fontSize:13, marginBottom:24}}>Fill in the details below. All fields are editable after saving too.</p>

          <div style={S.grid2}>
            <div>
              <label style={S.label}>Date</label>
              <input type="date" style={S.input} value={form.date} onChange={e=>setForm({...form,date:e.target.value})} />
            </div>
            <div>
              <label style={S.label}>Bank / Account Name</label>
              <input style={S.input} placeholder="e.g. Chase, HDFC..." value={form.bank_source} onChange={e=>setForm({...form,bank_source:e.target.value})} />
            </div>
          </div>

          <div style={{marginBottom:16}}>
            <label style={S.label}>Description / Merchant Name</label>
            <input style={S.input} placeholder="e.g. Starbucks, Netflix, Salary..." value={form.description} onChange={e=>setForm({...form,description:e.target.value})} />
          </div>

          <div style={S.grid3}>
            <div>
              <label style={S.label}>Amount</label>
              <input
                type="number"
                style={S.input}
                placeholder="e.g. -45.99 or 3500"
                value={form.amount}
                onChange={e=>setForm({...form,amount:e.target.value})}
              />
              <p style={{color:'#4a4a6a', fontSize:11, marginTop:4}}>Use − for expenses, + for income</p>
            </div>
            <div>
              <label style={S.label}>Currency</label>
              <select style={S.select} value={form.currency} onChange={e=>setForm({...form,currency:e.target.value})}>
                {CURRENCIES.map(c=><option key={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label style={S.label}>Category</label>
              <select style={S.select} value={form.category} onChange={e=>setForm({...form,category:e.target.value})}>
                {CATEGORIES.map(c=><option key={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <button onClick={handleManualSave} disabled={saving} style={{...S.saveBtn, opacity:saving?0.5:1}}>
            {saving ? 'Saving...' : '+ Save Transaction'}
          </button>

          {manualStatus && (
            <p style={S.status(manualStatus.startsWith('✓'))}>{manualStatus}</p>
          )}

          {recentManual.length > 0 && (
            <div style={S.tableWrap}>
              <div style={{padding:'14px 16px', borderBottom:'1px solid #1e1e2e', display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                <p style={{color:'#fff', fontWeight:600, fontSize:14}}>Recently Added</p>
                <span style={{color:'#4a4a6a', fontSize:12}}>{recentManual.length} records · click Edit to modify any field</span>
              </div>
              <table style={{width:'100%', borderCollapse:'collapse'}}>
                <thead>
                  <tr>
                    {['Date','Description','Amount','Currency','Category','Bank','Actions'].map(h=>(
                      <th key={h} style={S.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {recentManual.map(tx => renderEditableRow(tx, saveEdit))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
