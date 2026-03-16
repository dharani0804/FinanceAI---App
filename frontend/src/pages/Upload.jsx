import { useState } from 'react'

export default function Upload() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('')
  const [transactions, setTransactions] = useState([])
  const [loading, setLoading] = useState(false)
  const [drag, setDrag] = useState(false)

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setStatus('')
    setTransactions([])
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch('http://127.0.0.1:8000/upload', { method: 'POST', body: formData })
      const data = await res.json()
      if (data.success) {
        setStatus(`✓ ${data.transactions_imported} transactions imported`)
        setTransactions(data.transactions)
      } else {
        setStatus(`Error: ${data.error}`)
      }
    } catch (err) {
      setStatus(`Could not connect to backend`)
    } finally {
      setLoading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) setFile(dropped)
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-white mb-1">Upload Statement</h1>
        <p className="text-[#55556a] text-sm">Import your bank statement in CSV or PDF format</p>
      </div>

      <div
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        style={{border: `2px dashed ${drag ? '#3b82f6' : '#2a2a3a'}`}}
        className="bg-[#12121e] rounded-xl p-12 text-center mb-4 transition-all cursor-pointer"
        onClick={() => document.getElementById('fileInput').click()}
      >
        <div className="w-12 h-12 rounded-xl bg-[#1a1a28] flex items-center justify-center mx-auto mb-4 text-2xl">📎</div>
        <p className="text-white font-medium mb-1">
          {file ? file.name : 'Drop your file here'}
        </p>
        <p className="text-[#55556a] text-sm">
          {file ? `${(file.size/1024).toFixed(1)} KB · Ready to upload` : 'or click to browse · CSV or PDF'}
        </p>
        <input
          id="fileInput"
          type="file"
          accept=".pdf,.csv"
          className="hidden"
          onChange={e => setFile(e.target.files[0])}
        />
      </div>

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-[#1a1a28] disabled:text-[#55556a] disabled:cursor-not-allowed text-white font-medium py-3 rounded-xl transition-all mb-4"
      >
        {loading ? '⏳ Parsing statement...' : 'Upload & Import'}
      </button>

      {status && (
        <div style={{border:'1px solid #1a1a28'}} className="bg-[#12121e] rounded-xl px-4 py-3 mb-6">
          <p className={`text-sm font-medium ${status.startsWith('✓') ? 'text-emerald-400' : 'text-red-400'}`}>{status}</p>
        </div>
      )}

      {transactions.length > 0 && (
        <div style={{border:'1px solid #1a1a28'}} className="bg-[#12121e] rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-[#1a1a28]">
            <p className="text-white text-sm font-medium">Imported Transactions</p>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr style={{borderBottom:'1px solid #1a1a28'}}>
                <th className="text-left px-5 py-3 text-[#55556a] text-xs uppercase tracking-wider">Date</th>
                <th className="text-left px-5 py-3 text-[#55556a] text-xs uppercase tracking-wider">Description</th>
                <th className="text-right px-5 py-3 text-[#55556a] text-xs uppercase tracking-wider">Amount</th>
                <th className="text-left px-5 py-3 text-[#55556a] text-xs uppercase tracking-wider">Currency</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((t, i) => (
                <tr key={i} style={{borderBottom:'1px solid #1a1a28'}} className="hover:bg-[#1a1a28] transition-colors">
                  <td className="px-5 py-3 text-[#8888aa]">{t.date}</td>
                  <td className="px-5 py-3 text-white">{t.description}</td>
                  <td className={`px-5 py-3 text-right font-medium ${parseFloat(t.amount) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {parseFloat(t.amount) >= 0 ? '+' : ''}${Math.abs(parseFloat(t.amount)).toFixed(2)}
                  </td>
                  <td className="px-5 py-3 text-[#8888aa]">{t.currency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
