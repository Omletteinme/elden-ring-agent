import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

const EXAMPLE_QUESTIONS = [
  'What is the FP cost of Ancient Death Rancor?',
  'Compare the HP of Ancestor Spirit and Cemetery Shade',
  'How was Ancient Death Rancor changed in patch updates?',
  'What is the best armor set for a strength build?',
]

function Message({ role, content, searchTrace, isError }) {
  return (
    <div className={`message message-${role} ${isError ? 'message-error' : ''}`}>
      <div className="message-role">{role === 'user' ? 'You' : 'Agent'}</div>
      <div className="message-content">{content}</div>
      {searchTrace && searchTrace.length > 0 && (
        <details className="search-trace">
          <summary>{searchTrace.length} search{searchTrace.length > 1 ? 'es' : ''} made</summary>
          <ul>
            {searchTrace.map((s, i) => (
              <li key={i}><code>{s.query}</code></li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendQuestion(question) {
    if (!question.trim() || loading) return
    setMessages((m) => [...m, { role: 'user', content: question }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Request failed (${res.status})`)
      }
      const data = await res.json()
      setMessages((m) => [...m, { role: 'agent', content: data.answer, searchTrace: data.search_trace }])
    } catch (e) {
      setMessages((m) => [...m, { role: 'agent', content: `Something went wrong: ${e.message}. Is the backend running on ${API_URL}?`, isError: true }])
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    sendQuestion(input)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Elden Ring Agent</h1>
        <p>Answers grounded in an indexed wiki corpus (Weapons, Bosses, Talismans, Sorceries, Incantations) — not model memory. Every answer cites its sources.</p>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Try asking:</p>
            <div className="examples">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button key={q} className="example-chip" onClick={() => sendQuestion(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} {...m} />
        ))}
        {loading && (
          <div className="message message-agent">
            <div className="message-role">Agent</div>
            <div className="message-content loading-dots">Searching and thinking…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a weapon, boss, talisman, sorcery, or incantation…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Ask</button>
      </form>
    </div>
  )
}

export default App
