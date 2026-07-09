// LMTrade dashboard — polls the read-only JSON API and repaints every 5s.
const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
const time = (ts) => new Date(ts * 1000).toLocaleTimeString();

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}

function sideClass(s) { return s === "buy" ? "buy" : s === "sell" ? "sell" : "hold"; }

async function refresh() {
  try {
    const [s, trades, activity, logs, equity, strategies, realized] = await Promise.all([
      getJSON("/api/summary"),
      getJSON("/api/trades?limit=100"),
      getJSON("/api/activity?limit=100"),
      getJSON("/api/logs?limit=200"),
      getJSON("/api/equity"),
      getJSON("/api/leaderboard"),
      getJSON("/api/realized?limit=200"),
    ]);
    paintSummary(s);
    paintPositions(s.positions);
    paintTrades(trades);
    paintActivity(activity);
    paintLogs(logs);
    paintChart(equity);
    paintStrategies(strategies);
    paintRealized(realized);
  } catch (e) {
    console.error(e);
  }
}

function paintSummary(s) {
  const cur = s.currency || "EUR";
  $("curr").textContent = cur;
  const mode = $("mode");
  mode.textContent = s.mode;
  mode.className = "badge " + (s.mode === "live" ? "live" : "paper");

  const econ = s.economics || {};
  const net = econ.net_worth_eur != null ? econ.net_worth_eur : s.equity;
  const pnl = econ.pnl_eur != null ? econ.pnl_eur : net - (s.starting_cash || 0);
  $("net").textContent = fmt(net) + " " + cur;
  const pnlEl = $("pnl");
  pnlEl.textContent = (pnl >= 0 ? "▲ " : "▼ ") + fmt(pnl) + " " + cur + " P&L";
  pnlEl.className = "sub " + (pnl >= 0 ? "" : "");
  $("net").className = "v " + (pnl >= 0 ? "pos" : "neg");

  $("cash").textContent = fmt(s.cash) + " " + cur;
  $("reserve").textContent = fmt(s.reserve != null ? s.reserve : (econ.reserve_eur || 0)) + " " + cur;
  $("npos").textContent = s.num_positions;
  const compute = (econ.gpu_cost_accrued_usd || 0) + (econ.inference_cost_usd || 0);
  $("trcash").textContent = s.tr_account_cash != null ? fmt(s.tr_account_cash) + " " + cur : "—";

  const banner = $("econ");
  if (econ.self_sustaining) {
    banner.className = "econ-banner econ-ok";
    banner.innerHTML = `✅ <b>Self-sustaining</b> — gains cover all compute. Runway ${fmt(econ.runway_hours,1)}h at $${fmt(econ.gpu_usd_per_hour,3)}/hr GPU.`;
  } else if (econ.halt_trading) {
    banner.className = "econ-banner econ-warn";
    banner.innerHTML = `⛔ <b>Runway below floor</b> (${fmt(econ.runway_hours,1)}h) — new entries halted, managing exits only.`;
  } else {
    banner.className = "econ-banner econ-warn";
    banner.innerHTML = `⏳ <b>Subsidised</b> — not yet covering compute. Net worth $${fmt(econ.net_worth_usd,2)}, compute spent $${fmt(compute,4)}, runway ${fmt(econ.runway_hours,1)}h.`;
  }
}

function pnlCell(v) {
  if (v == null) return `<td class="muted">—</td>`;
  const cls = v >= 0 ? "pos" : "neg";
  const sign = v >= 0 ? "+" : "";
  return `<td class="num ${cls}">${sign}${fmt(v)}</td>`;
}

function paintPositions(rows) {
  $("positions").innerHTML = rows.length
    ? rows.map(p => `<tr><td>${p.symbol}</td><td><span class="kind">${p.kind || "equity"}</span></td>`
        + `<td>${p.isin || "—"}</td><td>${fmt(p.qty,4)}</td><td>${fmt(p.avg_price)}</td>`
        + `<td>${p.value == null ? "—" : fmt(p.value)}</td>${pnlCell(p.unrealized_pnl)}</tr>`).join("")
    : `<tr><td colspan="7" class="muted">No open positions.</td></tr>`;
}

function paintStrategies(rows) {
  $("strategies").innerHTML = rows && rows.length
    ? rows.map(g => {
        const params = Object.entries(g.params || {})
          .map(([k, v]) => `${k}=${typeof v === "number" ? Number(v).toFixed(2) : v}`).join(", ");
        return `<tr><td>${g.strategy}</td><td class="muted">${params}</td>`
          + `<td>${g.trades}</td>${pnlCell(g.pnl)}<td>${fmt(g.fitness, 4)}</td></tr>`;
      }).join("")
    : `<tr><td colspan="5" class="muted">No strategies evolved yet — they appear once trades close and the optimizer runs.</td></tr>`;
}

function paintRealized(r) {
  const total = (r && r.total_pnl) || 0;
  const card = $("realized");
  card.textContent = (total >= 0 ? "+" : "") + fmt(total);
  card.className = "v " + (total > 0 ? "pos" : total < 0 ? "neg" : "");
  $("realized-sub").textContent = r ? `${r.wins}W / ${r.losses}L closed` : "closed trades";
  const rows = (r && r.rows) || [];
  $("realized-rows").innerHTML = rows.length
    ? rows.map(o => `<tr><td>${o.closed_ts ? time(o.closed_ts) : "—"}</td><td>${o.symbol}</td>`
        + `<td><span class="kind">${o.kind}</span></td><td>${o.isin || "—"}</td>`
        + `<td>${fmt(o.entry_premium)}</td><td>${fmt(o.exit_premium)}</td>${pnlCell(o.pnl)}</tr>`).join("")
    : `<tr><td colspan="7" class="muted">No closed trades yet.</td></tr>`;
}

function paintTrades(rows) {
  $("trades").innerHTML = rows.length
    ? rows.map(t => `<tr><td>${time(t.ts)}</td><td>${t.symbol}</td>
        <td><span class="pill ${sideClass(t.side)}">${t.side}</span></td>
        <td>${fmt(t.qty,4)}</td><td>${fmt(t.price)}</td><td>${fmt(t.fee)}</td>
        <td>${t.confidence != null ? fmt(t.confidence) : "—"}</td></tr>`).join("")
    : `<tr><td colspan="7" class="muted">No trades yet.</td></tr>`;
}

function paintActivity(rows) {
  $("activity").innerHTML = rows.length
    ? rows.map(a => `<tr><td>${time(a.ts)}</td><td><span class="kind">${a.kind}</span></td>
        <td>${a.symbol || "—"}</td><td>${a.summary}</td></tr>`).join("")
    : `<tr><td colspan="4" class="muted">No activity yet.</td></tr>`;
}

function paintLogs(rows) {
  $("logs").innerHTML = rows.map(l =>
    `<div class="log-${l.level}">${time(l.ts)} [${l.level}] ${l.source || ""} ${l.message}</div>`
  ).join("") || `<div class="muted">No logs.</div>`;
}

function paintChart(curve) {
  const svg = $("chart");
  if (!curve || curve.length < 2) { svg.innerHTML = ""; return; }
  const vals = curve.map(p => p.equity);
  const min = Math.min(...vals), max = Math.max(...vals);
  const span = max - min || 1;
  const W = 600, H = 120, pad = 4;
  const pts = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * W;
    const y = H - pad - ((v - min) / span) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const up = vals[vals.length - 1] >= vals[0];
  const color = up ? "#3fb950" : "#f85149";
  svg.innerHTML = `
    <polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}" />
    <polygon fill="${color}22" points="0,${H} ${pts} ${W},${H}" />`;
}

// Tabs
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    ["positions", "realized", "strategies", "trades", "activity", "logs"].forEach(name =>
      $("tab-" + name).classList.toggle("hidden", name !== tab.dataset.tab));
  });
});

refresh();
setInterval(refresh, 5000);
