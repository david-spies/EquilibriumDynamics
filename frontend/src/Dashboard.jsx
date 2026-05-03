/**
 * EquilibriumDynamics — PINN Research Dashboard
 * Aesthetic: Scientific instrument meets dark-mode terminal
 * Stack: React + Recharts + WebSocket + Anthropic API
 */

import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

/* ─── Design tokens ─────────────────────────────────────────────────────── */
const T = {
  bg:      "#080E18",
  panel:   "#0D1829",
  border:  "#1A2E4A",
  teal:    "#00E5C0",
  amber:   "#F5A623",
  blue:    "#3D9EFF",
  purple:  "#9B6DFF",
  red:     "#FF5555",
  textPri: "#C8E0FF",
  textSec: "#5A7FA8",
  textDim: "#2A4060",
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: ${T.bg};
    color: ${T.textPri};
    font-family: 'Space Grotesk', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Scanline texture */
  body::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,229,192,0.012) 2px, rgba(0,229,192,0.012) 4px
    );
  }

  .app { position: relative; z-index: 1; display: flex; flex-direction: column; min-height: 100vh; }

  /* Header */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px;
    border-bottom: 1px solid ${T.border};
    background: ${T.panel};
    backdrop-filter: blur(12px);
  }
  .logo { display: flex; align-items: center; gap: 12px; }
  .logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: linear-gradient(135deg, ${T.teal}33, ${T.blue}33);
    border: 1px solid ${T.teal}66;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .logo-text { font-size: 17px; font-weight: 600; letter-spacing: -0.3px; }
  .logo-sub  { font-size: 11px; color: ${T.textSec}; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px; }
  .status-pill {
    display: flex; align-items: center; gap: 6px; padding: 4px 12px;
    border-radius: 100px; font-size: 11px; font-family: 'JetBrains Mono', monospace;
    border: 1px solid; letter-spacing: 0.5px;
  }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* Layout */
  .main { display: grid; grid-template-columns: 300px 1fr; gap: 0; flex: 1; }

  /* Sidebar */
  .sidebar {
    border-right: 1px solid ${T.border};
    background: ${T.panel};
    padding: 24px 20px;
    display: flex; flex-direction: column; gap: 24px;
    overflow-y: auto;
  }
  .section-label {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: ${T.textSec}; letter-spacing: 2px; text-transform: uppercase;
    padding-bottom: 8px; border-bottom: 1px solid ${T.border};
    margin-bottom: 4px;
  }

  /* Param sliders */
  .param-row { display: flex; flex-direction: column; gap: 6px; }
  .param-header { display: flex; justify-content: space-between; align-items: baseline; }
  .param-name { font-size: 13px; color: ${T.textPri}; font-weight: 500; }
  .param-val  { font-size: 13px; font-family: 'JetBrains Mono', monospace; color: ${T.teal}; }
  input[type=range] {
    -webkit-appearance: none; width: 100%; height: 2px;
    background: ${T.border}; border-radius: 2px; outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: ${T.teal}; cursor: pointer; border: 2px solid ${T.bg};
    box-shadow: 0 0 8px ${T.teal}66;
  }
  input[type=range]:hover { background: ${T.textDim}; }

  input[type=number] {
    width: 100%; background: ${T.bg}; border: 1px solid ${T.border};
    color: ${T.textPri}; padding: 7px 10px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; outline: none;
  }
  input[type=number]:focus { border-color: ${T.teal}88; }

  /* Buttons */
  .btn {
    width: 100%; padding: 10px; border-radius: 8px; border: none;
    font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all .15s; letter-spacing: 0.3px;
  }
  .btn-primary {
    background: linear-gradient(135deg, ${T.teal}22, ${T.blue}22);
    color: ${T.teal}; border: 1px solid ${T.teal}66;
  }
  .btn-primary:hover { background: linear-gradient(135deg, ${T.teal}33, ${T.blue}33); box-shadow: 0 0 16px ${T.teal}33; }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
  .btn-secondary {
    background: transparent; color: ${T.textSec}; border: 1px solid ${T.border};
    margin-top: 8px;
  }
  .btn-secondary:hover { border-color: ${T.textSec}; color: ${T.textPri}; }

  /* Content area */
  .content { padding: 24px; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }

  /* Metric row */
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .metric-card {
    background: ${T.panel}; border: 1px solid ${T.border};
    border-radius: 10px; padding: 14px 16px;
  }
  .metric-label { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: ${T.textSec}; letter-spacing: 1px; }
  .metric-val   { font-size: 24px; font-weight: 600; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

  /* Charts */
  .chart-card {
    background: ${T.panel}; border: 1px solid ${T.border};
    border-radius: 10px; padding: 20px;
  }
  .chart-title {
    font-size: 13px; font-weight: 600; color: ${T.textPri};
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
  }
  .chart-title .eq {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: ${T.textSec}; background: ${T.bg}; padding: 2px 8px;
    border-radius: 4px; border: 1px solid ${T.border};
  }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

  /* Loss stream */
  .loss-stream {
    height: 200px; overflow-y: auto; background: ${T.bg};
    border-radius: 6px; padding: 10px; font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: ${T.textSec}; border: 1px solid ${T.border};
  }
  .loss-line { padding: 2px 0; border-bottom: 1px solid ${T.textDim}; display: flex; gap: 12px; }
  .ls-step  { color: ${T.textDim}; min-width: 60px; }
  .ls-pde   { color: ${T.teal}; }
  .ls-bc    { color: ${T.amber}; }
  .ls-ic    { color: ${T.blue}; }

  /* Heatmap canvas */
  .heatmap-wrap { position: relative; }
  canvas.heatmap { width: 100%; border-radius: 6px; image-rendering: pixelated; display: block; }

  /* Dropzones */
  .dropzones { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .dropzone {
    border: 1.5px dashed ${T.border}; border-radius: 10px;
    padding: 20px; text-align: center; cursor: pointer;
    transition: all .2s; position: relative;
  }
  .dropzone:hover, .dropzone.drag { border-color: ${T.teal}; background: ${T.teal}08; }
  .dropzone .dz-icon { font-size: 24px; margin-bottom: 8px; }
  .dropzone .dz-label { font-size: 12px; color: ${T.textSec}; }
  .dropzone .dz-type  { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: ${T.textDim}; margin-top: 4px; }
  .dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

  /* AI analysis */
  .ai-panel {
    background: linear-gradient(135deg, ${T.purple}0A, ${T.bg});
    border: 1px solid ${T.purple}33; border-radius: 10px; padding: 20px;
  }
  .ai-output {
    min-height: 80px; font-size: 13px; line-height: 1.65; color: ${T.textPri};
    white-space: pre-wrap;
  }
  .ai-cursor { display: inline-block; width: 8px; height: 14px; background: ${T.purple}; animation: blink .8s infinite; vertical-align: text-bottom; }
  @keyframes blink { 0%,49%{opacity:1} 50%,100%{opacity:0} }

  /* Tooltip override */
  .recharts-tooltip-wrapper .recharts-default-tooltip {
    background: ${T.panel} !important; border: 1px solid ${T.border} !important;
    border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
  }

  /* Conservation badge */
  .conservation-ok   { color: ${T.teal}; }
  .conservation-warn { color: ${T.amber}; }
  .conservation-bad  { color: ${T.red};  }

  /* Responsive */
  @media (max-width: 900px) {
    .main      { grid-template-columns: 1fr; }
    .metrics   { grid-template-columns: 1fr 1fr; }
    .grid-2    { grid-template-columns: 1fr; }
    .dropzones { grid-template-columns: 1fr; }
  }
`;

/* ─── API helpers ─────────────────────────────────────────────────────────── */
const API = "http://localhost:8000";

async function apiFetch(path, opts = {}) {
  try {
    const r = await fetch(API + path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || r.statusText);
    }
    return r.json();
  } catch (err) {
    throw err;
  }
}

/* ─── Heatmap renderer ───────────────────────────────────────────────────── */
function drawHeatmap(canvas, data) {
  if (!canvas || !data) return;
  const { x, t, u_mean, u_std } = data;

  const nT = [...new Set(t)].length;
  const nX = [...new Set(x)].length;
  canvas.width  = nX;
  canvas.height = nT;

  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(nX, nT);

  const uMin = Math.min(...u_mean);
  const uMax = Math.max(...u_mean);
  const range = uMax - uMin || 1;

  for (let i = 0; i < u_mean.length; i++) {
    const norm = (u_mean[i] - uMin) / range; // 0..1
    const conf = Math.min(1, (u_std[i] * 20) || 0); // uncertainty band opacity
    // Teal→blue colour ramp
    const r = Math.round(0   + norm * 61);
    const g = Math.round(229 - norm * 71);
    const b = Math.round(192 + norm * 63);
    const idx = i * 4;
    img.data[idx]     = r;
    img.data[idx + 1] = g;
    img.data[idx + 2] = b;
    img.data[idx + 3] = 255 - Math.round(conf * 40);
  }
  ctx.putImageData(img, 0, 0);
}

/* ─── Streaming AI analysis via Anthropic API ────────────────────────────── */
async function streamAnalysis(params, metrics, onChunk) {
  const prompt = `You are a senior computational physicist reviewing PINN training results.

Physics parameters: v=${params.v} (convection velocity), D=${params.D} (diffusion coefficient).
Current metrics: PDE residual loss=${metrics.pdeLoss?.toFixed(4) ?? "N/A"}, 
                 BC loss=${metrics.bcLoss?.toFixed(4) ?? "N/A"},
                 IC loss=${metrics.icLoss?.toFixed(4) ?? "N/A"},
                 Max mass conservation error=${metrics.massErr?.toFixed(2) ?? "N/A"}%.

In 3-4 sentences, briefly interpret: (1) whether convergence is healthy, (2) the physical significance of the Péclet number Pe = v/D = ${(params.v / params.D).toFixed(1)}, and (3) one actionable suggestion to improve accuracy.`;

  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      stream: true,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  const reader = resp.body.getReader();
  const dec    = new TextDecoder();
  let buffer   = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += dec.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const json = line.slice(6).trim();
      if (json === "[DONE]") return;
      try {
        const evt = JSON.parse(json);
        if (evt.type === "content_block_delta" && evt.delta?.text) {
          onChunk(evt.delta.text);
        }
      } catch {}
    }
  }
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Main Dashboard Component                                                   */
/* ══════════════════════════════════════════════════════════════════════════ */
export default function EquilibriumDashboard() {
  // Physics params
  const [v, setV]     = useState(1.0);
  const [D, setD]     = useState(0.01);
  const [tSlice, setTSlice] = useState(0.5);
  const [adamIters, setAdamIters] = useState(15000);
  const [restore, setRestore]     = useState(false);

  // State
  const [training,    setTraining]    = useState(false);
  const [modelReady,  setModelReady]  = useState(false);
  const [lossHistory, setLossHistory] = useState([]);
  const [sliceData,   setSliceData]   = useState([]);
  const [heatmapData, setHeatmapData] = useState(null);
  const [conservation, setConservation] = useState(null);
  const [metrics, setMetrics]         = useState({ pdeLoss: null, bcLoss: null, icLoss: null, massErr: null });
  const [aiText,  setAiText]          = useState("");
  const [aiLoading, setAiLoading]     = useState(false);
  const [error,   setError]           = useState(null);
  const [wsStatus, setWsStatus]       = useState("disconnected");

  const wsRef      = useRef(null);
  const canvasRef  = useRef(null);
  const lossEndRef = useRef(null);

  const Pe = (v / D).toFixed(1);

  /* ── WebSocket connection with auto-reconnect ── */
  // reconnectRef holds the setTimeout id so we can cancel on unmount
  const reconnectRef = useRef(null);

  const connectWs = useCallback(() => {
    // Don't stack connections (StrictMode double-invoke guard)
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

    const ws = new WebSocket("ws://localhost:8000/ws/training");
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus("connected");
      // Clear any pending reconnect timer once we're successfully connected
      if (reconnectRef.current) { clearTimeout(reconnectRef.current); reconnectRef.current = null; }
    };

    ws.onclose = () => {
      setWsStatus("disconnected");
      // Auto-reconnect after 3 s unless the component is unmounting
      reconnectRef.current = setTimeout(() => connectWs(), 3000);
    };

    ws.onerror = () => {
      setWsStatus("error");
      ws.close(); // triggers onclose → reconnect
    };

    ws.onmessage = (e) => {
      const pkt = JSON.parse(e.data);
      if (pkt.event === "heartbeat") return; // backend keepalive — ignore in UI
      if (pkt.event === "training_complete") {
        setTraining(false);
        setModelReady(true);
        fetchSlice();
        return;
      }
      setLossHistory(prev => [...prev, pkt].slice(-200));
      setMetrics({
        pdeLoss: pkt.loss_pde,
        bcLoss:  pkt.loss_bc,
        icLoss:  pkt.loss_ic,
      });
    };
  }, []);

  useEffect(() => {
    connectWs();
    return () => {
      // On unmount: cancel reconnect timer and close socket cleanly
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => { lossEndRef.current?.scrollIntoView(); }, [lossHistory]);
  useEffect(() => { drawHeatmap(canvasRef.current, heatmapData); }, [heatmapData]);

  /* ── Actions ── */
  async function startTraining() {
    setError(null);
    setLossHistory([]);
    setTraining(true);
    setModelReady(false);
    try {
      await apiFetch("/train", {
        method: "POST",
        body: JSON.stringify({ v, D, adam_iters: adamIters, restore }),
      });
    } catch (err) {
      setError(err.message);
      setTraining(false);
    }
  }

  async function fetchSlice() {
    try {
      const r = await apiFetch("/predict/slice", {
        method: "POST",
        body: JSON.stringify({ v, D, t_val: tSlice, n_x: 200 }),
      });
      setSliceData(r.plot_data);
    } catch (err) { setError(err.message); }
  }

  async function fetchHeatmap() {
    try {
      const r = await apiFetch("/predict/heatmap", {
        method: "POST",
        body: JSON.stringify({ n_x: 80, n_t: 80, uq_samples: 30 }),
      });
      setHeatmapData(r);
    } catch (err) { setError(err.message); }
  }

  async function fetchConservation() {
    try {
      const r = await apiFetch("/predict/conservation");
      setConservation(r);
      setMetrics(m => ({ ...m, massErr: r.max_rel_err * 100 }));
    } catch (err) { setError(err.message); }
  }

  async function runAiAnalysis() {
    setAiText("");
    setAiLoading(true);
    try {
      await streamAnalysis({ v, D }, metrics, chunk => setAiText(t => t + chunk));
    } catch { setAiText("AI analysis unavailable — check API connectivity."); }
    setAiLoading(false);
  }

  function handleFileDrop(type) {
    return async (e) => {
      const file = e.target.files?.[0] ?? e.dataTransfer?.files?.[0];
      if (!file) return;
      const fd = new FormData();
      fd.append("file", file);
      await fetch(`${API}/upload/${type}`, { method: "POST", body: fd });
    };
  }

  /* ── Status pill ── */
  const statusColor = training ? T.amber : modelReady ? T.teal : T.textDim;
  const statusLabel = training ? "TRAINING" : modelReady ? "MODEL READY" : "IDLE";

  /* ── Loss log entries ── */
  const lastN = lossHistory.slice(-6);

  /* ── Conservation class ── */
  const massErrPct = metrics.massErr;
  const conservClass = massErrPct == null ? "" :
    massErrPct < 1  ? "conservation-ok" :
    massErrPct < 5  ? "conservation-warn" : "conservation-bad";

  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* ── Header ── */}
        <header className="header">
          <div className="logo">
            <div className="logo-icon">⚖</div>
            <div>
              <div className="logo-text">EquilibriumDynamics</div>
              <div className="logo-sub">PINN · 1D CONVECTION–DIFFUSION · CONSERVATION ENFORCED</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span className="mono" style={{ fontSize: 11, color: T.textSec }}>
              Pe = {Pe}
            </span>
            <div className="status-pill" style={{ borderColor: statusColor + "66", color: statusColor }}>
              <div className="status-dot" style={{ background: statusColor }} />
              {statusLabel}
            </div>
            <div className="status-pill" style={{ borderColor: T.textDim, color: T.textSec }}>
              WS: {wsStatus}
            </div>
          </div>
        </header>

        <div className="main">

          {/* ══ Sidebar ══ */}
          <aside className="sidebar">

            <div>
              <div className="section-label">Physical Parameters</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                <div className="param-row">
                  <div className="param-header">
                    <span className="param-name">Convection velocity <em style={{ fontStyle: "normal", color: T.textDim }}>v</em></span>
                    <span className="param-val">{v.toFixed(2)}</span>
                  </div>
                  <input type="range" min="0.1" max="5" step="0.05"
                    value={v} onChange={e => setV(+e.target.value)} />
                </div>

                <div className="param-row">
                  <div className="param-header">
                    <span className="param-name">Diffusion coeff. <em style={{ fontStyle: "normal", color: T.textDim }}>D</em></span>
                    <span className="param-val">{D.toFixed(3)}</span>
                  </div>
                  <input type="range" min="0.001" max="0.5" step="0.001"
                    value={D} onChange={e => setD(+e.target.value)} />
                </div>

                <div className="param-row">
                  <div className="param-header">
                    <span className="param-name">Time slice <em style={{ fontStyle: "normal", color: T.textDim }}>t</em></span>
                    <span className="param-val">{tSlice.toFixed(2)}</span>
                  </div>
                  <input type="range" min="0" max="1" step="0.01"
                    value={tSlice} onChange={e => setTSlice(+e.target.value)} />
                </div>

              </div>
            </div>

            <div>
              <div className="section-label">Training Config</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div>
                  <div className="param-name" style={{ fontSize: 12, marginBottom: 4 }}>Adam iterations</div>
                  <input type="number" value={adamIters} min={1000} max={100000} step={1000}
                    onChange={e => setAdamIters(+e.target.value)} />
                </div>
                <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: T.textSec, cursor: "pointer" }}>
                  <input type="checkbox" checked={restore} onChange={e => setRestore(e.target.checked)} />
                  Warm-start from checkpoint
                </label>
              </div>
            </div>

            <div>
              <div className="section-label">Controls</div>
              <button className="btn btn-primary" onClick={startTraining} disabled={training}>
                {training ? "Training…" : "▶  Train PINN"}
              </button>
              <button className="btn btn-secondary" onClick={fetchSlice} disabled={!modelReady}>
                Refresh slice
              </button>
              <button className="btn btn-secondary" onClick={fetchHeatmap} disabled={!modelReady}>
                Render heatmap + UQ
              </button>
              <button className="btn btn-secondary" onClick={fetchConservation} disabled={!modelReady}>
                Conservation diagnostic
              </button>
            </div>

            <div>
              <div className="section-label">Plug-and-Play Upload</div>
              <div className="dropzones">
                <label className="dropzone">
                  <div className="dz-icon">⚛</div>
                  <div className="dz-label">Drop physics</div>
                  <div className="dz-type">.py PDE file</div>
                  <input type="file" accept=".py" onChange={handleFileDrop("pde")} />
                </label>
                <label className="dropzone">
                  <div className="dz-icon">💾</div>
                  <div className="dz-label">Drop weights</div>
                  <div className="dz-type">.pt checkpoint</div>
                  <input type="file" accept=".pt" onChange={handleFileDrop("weights")} />
                </label>
              </div>
            </div>

            {error && (
              <div style={{ background: T.red + "18", border: `1px solid ${T.red}44`,
                borderRadius: 8, padding: "10px 12px", fontSize: 12, color: T.red }}>
                ⚠ {error}
              </div>
            )}

          </aside>

          {/* ══ Content ══ */}
          <main className="content">

            {/* Metric row */}
            <div className="metrics">
              <div className="metric-card">
                <div className="metric-label">PDE RESIDUAL</div>
                <div className="metric-val" style={{ color: T.teal, fontSize: 18 }}>
                  {metrics.pdeLoss != null ? metrics.pdeLoss.toExponential(2) : "—"}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">BC LOSS</div>
                <div className="metric-val" style={{ color: T.amber, fontSize: 18 }}>
                  {metrics.bcLoss != null ? metrics.bcLoss.toExponential(2) : "—"}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">IC LOSS</div>
                <div className="metric-val" style={{ color: T.blue, fontSize: 18 }}>
                  {metrics.icLoss != null ? metrics.icLoss.toExponential(2) : "—"}
                </div>
              </div>
              <div className="metric-card">
                <div className="metric-label">MASS CONSERVATION ERR</div>
                <div className={`metric-val ${conservClass}`} style={{ fontSize: 18 }}>
                  {massErrPct != null ? massErrPct.toFixed(2) + "%" : "—"}
                </div>
              </div>
            </div>

            <div className="grid-2">

              {/* Live convergence */}
              <div className="chart-card">
                <div className="chart-title">
                  Live convergence stream
                  <span className="eq">loss = R² + λ·BC + λ·IC</span>
                </div>
                {lossHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={lossHistory}>
                      <CartesianGrid strokeDasharray="2 4" stroke={T.border} />
                      <XAxis dataKey="step" tick={{ fill: T.textSec, fontSize: 10, fontFamily: "JetBrains Mono" }} />
                      <YAxis scale="log" domain={["auto","auto"]}
                        tick={{ fill: T.textSec, fontSize: 10, fontFamily: "JetBrains Mono" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="loss_pde" dot={false} stroke={T.teal}  strokeWidth={1.5} name="PDE" />
                      <Line type="monotone" dataKey="loss_bc"  dot={false} stroke={T.amber} strokeWidth={1.5} name="BC" />
                      <Line type="monotone" dataKey="loss_ic"  dot={false} stroke={T.blue}  strokeWidth={1.5} name="IC" />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center",
                    color: T.textDim, fontSize: 13, fontFamily: "JetBrains Mono" }}>
                    Waiting for training signal…
                  </div>
                )}

                {/* Loss log */}
                <div className="loss-stream" style={{ marginTop: 10 }}>
                  {lastN.map((p, i) => (
                    <div className="loss-line" key={i}>
                      <span className="ls-step">step {p.step}</span>
                      <span className="ls-pde">R²={p.loss_pde?.toExponential(2) ?? "—"}</span>
                      <span className="ls-bc">BC={p.loss_bc?.toExponential(2) ?? "—"}</span>
                      <span className="ls-ic">IC={p.loss_ic?.toExponential(2) ?? "—"}</span>
                    </div>
                  ))}
                  <div ref={lossEndRef} />
                </div>
              </div>

              {/* Spatial slice */}
              <div className="chart-card">
                <div className="chart-title">
                  Solution slice <span className="eq">u(x, t={tSlice.toFixed(2)})</span>
                </div>
                {sliceData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={sliceData}>
                      <CartesianGrid strokeDasharray="2 4" stroke={T.border} />
                      <XAxis dataKey="x" tickCount={6}
                        tick={{ fill: T.textSec, fontSize: 10, fontFamily: "JetBrains Mono" }}
                        label={{ value: "x", position: "insideBottom", offset: -2, fill: T.textSec, fontSize: 11 }} />
                      <YAxis tick={{ fill: T.textSec, fontSize: 10, fontFamily: "JetBrains Mono" }}
                        label={{ value: "u", angle: -90, position: "insideLeft", fill: T.textSec, fontSize: 11 }} />
                      <Tooltip />
                      <ReferenceLine y={0} stroke={T.textDim} strokeDasharray="4 4" />
                      <Line type="monotone" dataKey="u" dot={false} stroke={T.teal} strokeWidth={2} name="u(x,t)" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 260, display: "flex", alignItems: "center", justifyContent: "center",
                    color: T.textDim, fontSize: 13, fontFamily: "JetBrains Mono" }}>
                    Run inference to see solution slice
                  </div>
                )}
              </div>
            </div>

            {/* Heatmap + conservation */}
            <div className="grid-2">
              <div className="chart-card">
                <div className="chart-title">
                  u(x, t) field — heatmap + UQ overlay
                  <span className="eq">x→ t↑</span>
                </div>
                <div className="heatmap-wrap">
                  <canvas ref={canvasRef} className="heatmap"
                    style={{ height: 200, background: T.bg, borderRadius: 6 }} />
                  {!heatmapData && (
                    <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center",
                      justifyContent: "center", color: T.textDim, fontSize: 13, fontFamily: "JetBrains Mono" }}>
                      Click "Render heatmap + UQ"
                    </div>
                  )}
                </div>
                {heatmapData && (
                  <div style={{ fontSize: 11, color: T.textSec, marginTop: 8, fontFamily: "JetBrains Mono" }}>
                    Colour: u mean · Opacity: ±2σ confidence · {heatmapData.u_mean?.length} grid points
                  </div>
                )}
              </div>

              {/* Conservation */}
              <div className="chart-card">
                <div className="chart-title">
                  Mass conservation ∫u dx vs. t
                </div>
                {conservation ? (
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={conservation.t.map((ti, i) => ({
                      t: ti.toFixed(2),
                      mass: +conservation.mass[i].toFixed(5),
                      err:  +(conservation.rel_error[i] * 100).toFixed(4),
                    }))}>
                      <CartesianGrid strokeDasharray="2 4" stroke={T.border} />
                      <XAxis dataKey="t" tick={{ fill: T.textSec, fontSize: 10 }} />
                      <YAxis yAxisId="mass" tick={{ fill: T.teal, fontSize: 10 }} />
                      <YAxis yAxisId="err" orientation="right" tick={{ fill: T.amber, fontSize: 10 }} />
                      <Tooltip />
                      <Line yAxisId="mass" dataKey="mass" dot={false} stroke={T.teal}  strokeWidth={2} name="∫u dx" />
                      <Line yAxisId="err"  dataKey="err"  dot={false} stroke={T.amber} strokeWidth={1.5} strokeDasharray="4 2" name="Rel. err %" />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center",
                    color: T.textDim, fontSize: 13, fontFamily: "JetBrains Mono" }}>
                    Run conservation diagnostic
                  </div>
                )}
              </div>
            </div>

            {/* AI Analysis */}
            <div className="ai-panel">
              <div className="chart-title" style={{ marginBottom: 12 }}>
                <span style={{ color: T.purple }}>◆</span>
                AI Physics Interpretation
                <button className="btn btn-secondary" style={{ width: "auto", padding: "4px 14px", marginLeft: "auto", fontSize: 12 }}
                  onClick={runAiAnalysis} disabled={aiLoading}>
                  {aiLoading ? "Analysing…" : "Analyse current state"}
                </button>
              </div>
              <div className="ai-output">
                {aiText || <span style={{ color: T.textDim, fontFamily: "JetBrains Mono", fontSize: 12 }}>
                  Click "Analyse current state" for real-time AI interpretation of the PDE residuals and Péclet number (Pe = {Pe}).
                </span>}
                {aiLoading && <span className="ai-cursor" />}
              </div>
            </div>

          </main>
        </div>
      </div>
    </>
  );
}
