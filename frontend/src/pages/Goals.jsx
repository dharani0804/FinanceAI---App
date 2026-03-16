import { useState } from 'react'

export default function Goals() {
  const [goals, setGoals] = useState([])
  const [form, setForm] = useState({ title: '', target: '', currency: 'USD' })

  const addGoal = () => {
    if (!form.title || !form.target) return
    setGoals(prev => [...prev, { ...form, id: Date.now(), current: 0 }])
    setForm({ title: '', target: '', currency: 'USD' })
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-cyan-400 mb-6">Financial Goals</h1>
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800 max-w-lg mb-6">
        <h2 className="text-white font-semibold mb-4">Add New Goal</h2>
        <input
          placeholder="Goal title"
          value={form.title}
          onChange={e => setForm({...form, title: e.target.value})}
          className="w-full bg-gray-800 text-white rounded-lg px-4 py-2 text-sm mb-3 outline-none"
        />
        <div className="flex gap-2 mb-3">
          <input
            placeholder="Target amount"
            type="number"
            value={form.target}
            onChange={e => setForm({...form, target: e.target.value})}
            className="flex-1 bg-gray-800 text-white rounded-lg px-4 py-2 text-sm outline-none"
          />
          <select
            value={form.currency}
            onChange={e => setForm({...form, currency: e.target.value})}
            className="bg-gray-800 text-white rounded-lg px-3 py-2 text-sm outline-none"
          >
            <option>USD</option>
            <option>EUR</option>
            <option>GBP</option>
            <option>INR</option>
            <option>AUD</option>
          </select>
        </div>
        <button
          onClick={addGoal}
          className="w-full bg-cyan-500 text-white font-semibold py-2 rounded-lg"
        >
          Add Goal
        </button>
      </div>
      <div className="space-y-3 max-w-lg">
        {goals.length === 0 && (
          <p className="text-gray-500 text-sm">No goals yet — add one above!</p>
        )}
        {goals.map(goal => (
          <div key={goal.id} className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <div className="flex justify-between items-center mb-2">
              <p className="text-white font-semibold">{goal.title}</p>
              <p className="text-cyan-400 text-sm">{goal.currency} {goal.target}</p>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div className="bg-cyan-500 h-2 rounded-full w-0"></div>
            </div>
            <p className="text-gray-500 text-xs mt-1">0% complete</p>
          </div>
        ))}
      </div>
    </div>
  )
}
