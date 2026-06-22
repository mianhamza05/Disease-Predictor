import { useState, useEffect } from 'react'
import './index.css'

const API = 'http://127.0.0.1:8000'

const SEVERITY_CONFIG = {
  mild:     { label: 'Mild',     color: '#68d391', bg: 'rgba(104,211,145,0.12)' },
  moderate: { label: 'Moderate', color: '#f6e05e', bg: 'rgba(246,224,94,0.12)'  },
  severe:   { label: 'Severe',   color: '#fc8181', bg: 'rgba(252,129,129,0.12)' },
  critical: { label: 'Critical', color: '#fc8181', bg: 'rgba(252,0,0,0.18)'     },
  unknown:  { label: 'Unknown',  color: '#8899aa', bg: 'rgba(136,153,170,0.12)' },
}

function SymptomTag({ symptom, selected, onClick }) {
  const label = symptom.replace(/_/g, ' ')
  return (
    <button onClick={() => onClick(symptom)} style={{
      padding: '6px 14px', borderRadius: '999px',
      border: selected ? '1.5px solid var(--accent)' : '1.5px solid var(--border)',
      background: selected ? 'rgba(99,179,237,0.15)' : 'transparent',
      color: selected ? 'var(--accent)' : 'var(--text-muted)',
      fontSize: '13px', cursor: 'pointer', transition: 'all 0.18s',
      fontFamily: 'DM Sans, sans-serif', whiteSpace: 'nowrap', userSelect: 'none',
    }}>
      {selected && '✓ '}{label}
    </button>
  )
}

function ConfBar({ disease, confidence, rank }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: rank === 0 ? 'var(--accent2)' : 'var(--text-muted)' }}>
          {rank === 0 && '★ '}{disease}
        </span>
        <span style={{ fontSize: 13, color: 'var(--text)', fontWeight: 600 }}>{confidence}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.07)', overflow: 'hidden' }}>
        <div style={{
          height: '100%', borderRadius: 3, width: `${confidence}%`,
          background: rank === 0 ? 'linear-gradient(90deg, var(--accent), var(--accent2))' : 'rgba(255,255,255,0.2)',
          transition: 'width 0.8s ease',
        }} />
      </div>
    </div>
  )
}

function ResultCard({ result }) {
  const sev = SEVERITY_CONFIG[result.severity] || SEVERITY_CONFIG.unknown
  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: 28, animation: 'fadeUp 0.4s ease' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
        <div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Predicted Diagnosis</p>
          <h2 style={{ fontSize: 26, fontFamily: 'Syne, sans-serif', fontWeight: 700, color: 'var(--accent2)' }}>{result.disease}</h2>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <div style={{ padding: '6px 14px', borderRadius: 999, background: sev.bg, border: `1px solid ${sev.color}44`, color: sev.color, fontSize: 13, fontWeight: 600 }}>{sev.label}</div>
          <div style={{ padding: '6px 14px', borderRadius: 999, background: 'rgba(99,179,237,0.12)', border: '1px solid rgba(99,179,237,0.3)', color: 'var(--accent)', fontSize: 13, fontWeight: 700 }}>{result.confidence}% confidence</div>
        </div>
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: 14, lineHeight: 1.7, marginBottom: 24, borderLeft: '3px solid var(--border)', paddingLeft: 14 }}>{result.description}</p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div>
          <h3 style={{ fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>Precautions</h3>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {result.precautions.map((p, i) => (
              <li key={i} style={{ display: 'flex', gap: 10, fontSize: 13, color: 'var(--text)' }}>
                <span style={{ color: 'var(--green)', flexShrink: 0 }}>→</span> {p}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 style={{ fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>Top Matches</h3>
          {result.top_predictions.map((p, i) => <ConfBar key={i} disease={p.disease} confidence={p.confidence} rank={i} />)}
        </div>
      </div>
      <div style={{ marginTop: 24, padding: '12px 16px', borderRadius: 'var(--radius-sm)', background: 'rgba(252,129,129,0.07)', border: '1px solid rgba(252,129,129,0.2)', fontSize: 12, color: 'var(--red)', lineHeight: 1.6 }}>
        ⚠️ This is an AI-based prediction tool for educational purposes only. Always consult a qualified medical professional for diagnosis and treatment.
      </div>
    </div>
  )
}

export default function App() {
  const [allSymptoms, setAllSymptoms] = useState([])
  const [selected, setSelected] = useState([])
  const [search, setSearch] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    const checkApi = async () => {
      try {
        const res = await fetch(`${API}/health`)
        if (res.ok) {
          setApiStatus('online')
          const data = await res.json()
          setAllSymptoms(data.symptoms || [])
        } else {
          setApiStatus('offline')
        }
      } catch {
        setApiStatus('offline')
      }
    }
    checkApi()
  }, [])

  const toggleSymptom = (s) => {
    setSelected(prev => prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s])
  }

  const predict = async () => {
    if (selected.length === 0) {
      setError('Select at least one symptom')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: selected })
      })
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError('Prediction failed: ' + e.message)
    }
    setLoading(false)
  }

  const filtered = allSymptoms.filter(s => s.replace(/_/g, ' ').toLowerCase().includes(search.toLowerCase()))

  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh', padding: '40px 20px' }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ marginBottom: 40, animation: 'fadeUp 0.5s ease' }}>
          <h1 style={{ fontSize: 36, fontFamily: 'Syne, sans-serif', fontWeight: 800, marginBottom: 8, color: 'var(--text)' }}>Disease Predictor</h1>
          <p style={{ fontSize: 14, color: 'var(--text-muted)' }}>Select your symptoms and get AI-powered disease predictions</p>
          {apiStatus === 'offline' && <div style={{ marginTop: 12, padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'rgba(252,129,129,0.1)', border: '1px solid rgba(252,129,129,0.3)', color: 'var(--red)', fontSize: 13 }}>🔴 API Offline — start the backend</div>}
        </div>

        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: 24, marginBottom: 24, animation: 'fadeUp 0.5s ease 0.1s both' }}>
          <h2 style={{ fontSize: 14, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>Search Symptoms</h2>
          <input type="text" placeholder="Search symptoms..." value={search} onChange={e => setSearch(e.target.value)} style={{ width: '100%', padding: '10px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', background: 'rgba(0,0,0,0.2)', color: 'var(--text)', fontSize: 14, marginBottom: 16 }} />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {filtered.slice(0, 20).map(s => <SymptomTag key={s} symptom={s} selected={selected.includes(s)} onClick={toggleSymptom} />)}
          </div>
        </div>

        <button onClick={predict} disabled={loading || apiStatus === 'offline'} style={{ width: '100%', padding: '12px 24px', borderRadius: 'var(--radius)', border: 'none', background: loading || apiStatus === 'offline' ? 'rgba(99,179,237,0.3)' : 'linear-gradient(135deg, var(--accent), var(--accent2))', color: 'white', fontSize: 15, fontWeight: 600, cursor: loading || apiStatus === 'offline' ? 'not-allowed' : 'pointer', transition: 'all 0.3s', marginBottom: 24 }}>
          {loading ? '🔄 Predicting...' : 'Get Prediction'}
        </button>

        {error && <div style={{ padding: 16, borderRadius: 'var(--radius)', background: 'rgba(252,129,129,0.1)', border: '1px solid rgba(252,129,129,0.3)', color: 'var(--red)', fontSize: 14, marginBottom: 24 }}>{error}</div>}
        {result && <ResultCard result={result} />}
      </div>
    </div>
  )
}