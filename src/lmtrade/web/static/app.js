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
    const [s, trades, activity, logs, equity, strategies, realized, signals, news, analysis, ctrl] =
      await Promise.all([
        getJSON("/api/summary"),
        getJSON("/api/trades?limit=100"),
        getJSON("/api/activity?limit=100"),
        getJSON("/api/logs?limit=200"),
        getJSON("/api/equity"),
        getJSON("/api/leaderboard"),
        getJSON("/api/realized?limit=200"),
        getJSON("/api/signals?min_confidence=0.6"),
        getJSON("/api/news"),
        getJSON("/api/analysis"),
        getJSON("/api/control"),
      ]);
    window._ctrl = ctrl;
    paintControl(ctrl, s);
    paintSummary(s);
    paintPositions(s.positions);
    paintTrades(trades);
    paintActivity(activity);
    paintLogs(logs);
    paintChart(equity);
    paintStrategies(strategies);
    paintRealized(realized);
    paintSignals(signals);
    paintNews(news);
    paintAnalysis(analysis);
  } catch (e) {
    console.error(e);
  }
}

function paintControl(ctrl, s) {
  const isLive = ctrl.mode === "live";
  const badge = $("mode");
  badge.textContent = ctrl.mode;
  badge.className = "badge " + (isLive ? "live" : "paper");
  $("armwrap").classList.toggle("show", isLive);
  $("arm-toggle").checked = !!ctrl.armed;
  $("armed-flag").classList.toggle("show", !!ctrl.armed);

  // Second guard: only relevant when armed AND the balance is low.
  const showGuard2 = isLive && ctrl.armed && ctrl.low_balance;
  $("guard2").classList.toggle("show", showGuard2);
  $("arm2-toggle").checked = !!ctrl.double_armed;

  // IS / IS-NOT executing real orders banner — the single most important
  // status in live mode. Hidden entirely in paper mode.
  const b = $("exec-banner");
  b.classList.toggle("show", isLive);
  if (isLive) {
    if (ctrl.executing) {
      b.className = "exec-banner show exec-live";
      $("exec-ic").textContent = "🔴";
      $("exec-text").innerHTML = "<b>EXECUTING REAL ORDERS</b> — live Trade Republic "
        + "fills with real money. Disarm to stop.";
    } else {
      b.className = "exec-banner show exec-sim";
      $("exec-ic").textContent = "🛡️";
      const why = !ctrl.armed
        ? "not armed — simulating on paper"
        : ctrl.low_balance
          ? `net worth under €${ctrl.low_balance_threshold} — low-balance guard not armed`
          : "simulating";
      $("exec-text").innerHTML = `<b>NOT executing real orders</b> (${why}).`;
    }
  }
}

function paintSummary(s) {
  const cur = s.currency || "EUR";
  $("curr").textContent = cur;
  const mode = $("mode");
  // mode text/class is owned by paintControl; leave it here as a fallback.
  if (!window._ctrl) { mode.textContent = s.mode; mode.className = "badge " + (s.mode === "live" ? "live" : "paper"); }

  const econ = s.economics || {};
  const ctrl = window._ctrl || {};
  const isLive = ctrl.mode === "live";
  const posValue = (s.positions || []).reduce((a, p) => a + (p.value || 0), 0);

  // In LIVE mode the headline figures are the REAL Trade Republic account —
  // cash from the live balance, net worth = real cash + open position value,
  // P&L vs the baseline captured when live began. Shown as "—" (never the
  // simulated paper 100) until the real balance is actually fetched. In paper
  // mode they're the simulated book's economics.
  let net, cash, pnl, netKnown;
  if (isLive) {
    cash = s.tr_account_cash;
    net = cash != null ? cash + posValue : null;
    netKnown = net != null;
    const base = s.tr_baseline_net_worth != null ? s.tr_baseline_net_worth : cash;
    pnl = netKnown ? net - base : null;
  } else {
    cash = s.cash;
    net = econ.net_worth_eur != null ? econ.net_worth_eur : s.equity;
    netKnown = true;
    pnl = econ.pnl_eur != null ? econ.pnl_eur : net - (s.starting_cash || 0);
  }

  $("net").textContent = netKnown ? fmt(net) + " " + cur : "—";
  const pnlEl = $("pnl");
  pnlEl.className = "sub";
  if (pnl != null) {
    pnlEl.textContent = (pnl >= 0 ? "▲ " : "▼ ") + fmt(pnl) + " " + cur + " P&L";
  } else {
    pnlEl.textContent = isLive ? "awaiting live TR balance…" : "—";
  }
  // Net worth is red while mark-to-market sits below the last REALIZED net
  // worth (locked in at the most recent closed trade), green at or above.
  const mark = s.last_realized_net_worth != null ? s.last_realized_net_worth : (s.starting_cash || 0);
  $("net").className = "v " + (!netKnown ? "" : net < mark - 1e-9 ? "neg" : "pos");

  const cashEl = $("cash");
  cashEl.textContent = cash != null ? fmt(cash) + " " + cur : "—";
  cashEl.className = "v" + (isLive && ctrl.low_balance ? " warn" : "");
  $("reserve").textContent = fmt(s.reserve != null ? s.reserve : (econ.reserve_eur || 0)) + " " + cur;
  $("npos").textContent = s.num_positions;
  const compute = (econ.gpu_cost_accrued_usd || 0) + (econ.inference_cost_usd || 0);
  const banner = $("econ");
  if (econ.self_sustaining) {
    banner.className = "econ-banner econ-ok";
    banner.innerHTML = `✅ <b>Self-sustaining</b> — covering its own compute cost. Runway ${fmt(econ.runway_hours,1)}h.`;
  } else if (econ.halt_trading) {
    banner.className = "econ-banner econ-warn";
    banner.innerHTML = `⛔ <b>Runway below floor</b> (${fmt(econ.runway_hours,1)}h) — new entries halted, managing exits only.`;
  } else {
    banner.className = "econ-banner econ-warn";
    banner.innerHTML = `⏳ <b>Subsidised</b> — not yet covering compute. Net worth $${fmt(econ.net_worth_usd,2)}, compute spent $${fmt(compute,4)}, runway ${fmt(econ.runway_hours,1)}h.`;
  }

  // Provider warning banner (e.g. Claude usage limit hit)
  const pwBanner = $("provider-warning-banner");
  const pw = s.provider_warnings;
  if (pw && pw.msg) {
    pwBanner.innerHTML = `⚠️ <b>Research provider warning:</b> ${pw.msg}
      <button onclick="dismissProviderWarning()" style="margin-left:12px;padding:2px 8px;cursor:pointer;border-radius:4px;border:none;background:#555;color:#fff;font-size:12px">Dismiss</button>`;
    pwBanner.classList.add("show");
  } else {
    pwBanner.classList.remove("show");
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
    ? rows.map(p => {
        const pending = p.status === "pending";
        const symbol = pending
          ? `${p.symbol} <span class="pill hold" title="Order submitted to Trade Republic, awaiting confirmation">PENDING</span>`
          : p.symbol;
        const valueCell = pending
          ? `<td class="muted">pending…</td>`
          : `<td>${p.value == null ? "—" : fmt(p.value)}</td>`;
        const pnlCellHtml = pending ? `<td class="muted">pending…</td>` : pnlCell(p.unrealized_pnl);
        return `<tr><td>${symbol}</td><td><span class="kind">${p.kind || "equity"}</span></td>`
          + `<td>${p.isin || "—"}</td><td>${fmt(p.qty,4)}</td><td>${fmt(p.avg_price)}</td>`
          + `<td>${p.sl_premium == null ? "—" : fmt(p.sl_premium)}</td>`
          + `<td>${p.tp_premium == null ? "—" : fmt(p.tp_premium)}</td>`
          + valueCell + pnlCellHtml + `</tr>`;
      }).join("")
    : `<tr><td colspan="9" class="muted">No open positions.</td></tr>`;
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

  // Unread badge: number of realized closes the user hasn't looked at. Total
  // closed count is wins+losses; we persist the last-seen count and show the
  // delta until the Realized tab is opened.
  const closedCount = r ? (r.wins || 0) + (r.losses || 0) : 0;
  const seen = Number(localStorage.getItem("realizedSeen") || 0);
  const badge = $("realized-unread");
  const active = document.querySelector('.tab[data-tab="realized"]').classList.contains("active");
  if (active) {
    localStorage.setItem("realizedSeen", String(closedCount));
    badge.classList.add("hidden");
  } else {
    const unread = Math.max(0, closedCount - seen);
    badge.textContent = unread > 99 ? "99+" : String(unread);
    badge.classList.toggle("hidden", unread === 0);
  }
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

function sentClass(s) {
  return s === "bullish" ? "buy" : s === "bearish" ? "sell" : "hold";
}

function paintSignals(rows) {
  $("signals").innerHTML = rows && rows.length
    ? rows.map(r => {
        const cause = (r.signals || [])
          .filter(s => s.direction && s.direction !== "hold")
          .map(s => `<span class="pill ${sideClass(s.direction)}">${s.provider} ${fmt(s.confidence)}</span> ${s.rationale || ""}`)
          .join("<br>") || `<span class="muted">${r.rationale || "—"}</span>`;
        return `<tr><td>${r.symbol}</td><td><span class="pill ${sideClass(r.direction)}">${r.direction}</span></td>`
          + `<td>${fmt(r.confidence)}</td><td>${cause}</td></tr>`;
      }).join("")
    : `<tr><td colspan="4" class="muted">No strong signals right now (needs confidence ≥ 0.6).</td></tr>`;
}

function paintNews(rows) {
  $("news").innerHTML = rows && rows.length
    ? rows.map(n => `<tr><td>${n.ts ? time(n.ts) : "—"}</td><td>${n.symbol}</td>`
        + `<td><span class="pill ${sentClass(n.sentiment)}">${n.sentiment || "—"}</span></td>`
        + `<td>${(n.text || "").slice(0, 240)}</td></tr>`).join("")
    : `<tr><td colspan="4" class="muted">No news yet.</td></tr>`;
}

function paintAnalysis(a) {
  const syms = (a && a.symbols) || {};
  const keys = Object.keys(syms);
  const meta = $("analysis-meta");
  if (a && a.ts) {
    const age = ((Date.now() / 1000 - a.ts) / 3600).toFixed(1);
    meta.textContent = `Compiled ${age}h ago · valid ${a.valid_hours || 24}h · ${keys.length} symbols`;
  } else {
    meta.textContent = "";
  }
  $("analysis").innerHTML = keys.length
    ? keys.map(k => {
        const v = syms[k] || {};
        return `<tr><td>${k}</td><td><span class="pill ${sentClass(v.bias)}">${v.bias || "—"}</span></td>`
          + `<td>${fmt(v.confidence)}</td><td class="muted">${v.notes || ""}</td></tr>`;
      }).join("")
    : `<tr><td colspan="4" class="muted">No daily analysis compiled yet.</td></tr>`;
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
    ["positions", "realized", "signals", "news", "analysis", "strategies", "trades", "activity", "logs"].forEach(name =>
      $("tab-" + name).classList.toggle("hidden", name !== tab.dataset.tab));
  });
});

async function postJSON(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

// Paper/live toggle — click the badge.
$("mode").addEventListener("click", async () => {
  const ctrl = window._ctrl || {mode: "paper"};
  const next = ctrl.mode === "live" ? "paper" : "live";
  if (next === "live" &&
      !confirm("Switch to LIVE mode?\n\nThis shows your real Trade Republic " +
               "account. Execution stays SIMULATED until you separately arm it.")) return;
  await postJSON("/api/control/mode", {mode: next});
  refresh();
});

// Arm / disarm real live execution.
$("arm-toggle").addEventListener("change", async (e) => {
  const ctrl = window._ctrl || {};
  if (!e.target.checked) {
    await postJSON("/api/control/disarm", {});
    refresh();
    return;
  }
  // First guard: hard confirmation. Below the fee-drag threshold this arms
  // but does NOT execute yet — the second guard (below) is required.
  const nw = ctrl.net_worth != null ? ctrl.net_worth.toFixed(2) + " EUR" : "unknown";
  if (!confirm("⚠ ARM REAL LIVE EXECUTION ⚠\n\nThe bot will place REAL Trade " +
               "Republic orders with REAL money, against TR's ToS. Fills are " +
               "irreversible.\n\nLive net worth: " + nw + "\n\nProceed?")) {
    e.target.checked = false; return;
  }
  const res = await postJSON("/api/control/arm", {confirm: true});
  if (!res.armed) e.target.checked = false;
  refresh();
});

// Second guard: allow real orders while net worth is under the threshold.
$("arm2-toggle").addEventListener("change", async (e) => {
  const ctrl = window._ctrl || {};
  if (!e.target.checked) {
    await postJSON("/api/control/double-disarm", {});
    refresh();
    return;
  }
  const thr = ctrl.low_balance_threshold || 100;
  const nw = ctrl.net_worth != null ? ctrl.net_worth.toFixed(2) + " EUR" : "unknown";
  if (!confirm("⚠ LOW-BALANCE EXECUTION ⚠\n\nNet worth (" + nw + ") is under €" +
               thr + ". At this size the flat ~1 EUR Trade Republic fee is more " +
               "than 10% of a typical order — a severe drag that is STRONGLY " +
               "ADVISED AGAINST.\n\nArm the second guard to execute real orders " +
               "anyway?")) {
    e.target.checked = false; return;
  }
  const res = await postJSON("/api/control/double-arm", {confirm: true});
  if (!res.double_armed) e.target.checked = false;
  refresh();
});

// ------------------------------------------------------------------ settings
const SECTION_LABELS = {
  general: "General", loop: "Loop", options: "Options", risk: "Risk",
  economics: "Economics", research: "Research", tr: "Trade Republic",
  data: "Data", model: "Model",
};

function fieldInput(f) {
  const id = "set_" + f.key.replace(/\./g, "_");
  if (f.kind === "bool") {
    return `<input type="checkbox" id="${id}" data-key="${f.key}" data-kind="bool" ${f.value ? "checked" : ""}>`;
  }
  if (f.kind === "select") {
    const opts = (f.options || []).map(o =>
      `<option value="${o}" ${String(f.value) === o ? "selected" : ""}>${o}</option>`).join("");
    return `<select id="${id}" data-key="${f.key}" data-kind="select">${opts}</select>`;
  }
  const type = (f.kind === "number" || f.kind === "int") ? "number" : "text";
  const step = f.kind === "int" ? "1" : "any";
  return `<input type="${type}" step="${step}" id="${id}" data-key="${f.key}" data-kind="${f.kind}" value="${f.value}">`;
}

async function openSettings() {
  const data = await getJSON("/api/settings");
  $("settings-note").textContent = data.note || "";
  const bySection = {};
  data.fields.forEach(f => { (bySection[f.section] = bySection[f.section] || []).push(f); });
  let html = "";
  Object.keys(bySection).forEach(sec => {
    html += `<div class="setsection">${SECTION_LABELS[sec] || sec}</div>`;
    bySection[sec].forEach(f => {
      html += `<div class="setrow"><label>${f.field}</label>${fieldInput(f)}</div>`;
    });
  });
  $("settings-fields").innerHTML = html;
  $("settings-status").textContent = "";
  $("settings-overlay").classList.add("show");
}

async function saveSettings() {
  const changes = {};
  document.querySelectorAll("#settings-fields [data-key]").forEach(el => {
    changes[el.dataset.key] = el.dataset.kind === "bool" ? el.checked : el.value;
  });
  $("settings-status").textContent = "Saving…";
  const res = await postJSON("/api/settings", {changes});
  const n = Object.keys(res.applied || {}).length;
  const bad = (res.rejected || []).length;
  if (res.restarting) {
    $("settings-overlay").classList.remove("show");
    showRestartOverlay();
  } else {
    $("settings-status").textContent =
      `Saved ${n} setting${n === 1 ? "" : "s"}${bad ? `, ${bad} rejected` : ""}` +
      (n > 0 ? " — restart `lmtrade run` to apply." : ".");
  }
}

function showRestartOverlay() {
  const el = document.createElement("div");
  el.id = "restart-overlay";
  el.className = "restart-overlay";
  el.innerHTML =
    `<div class="r-title">⟳ Restarting engine…</div>` +
    `<div class="r-sub">Applying new settings, back in a moment.</div>`;
  document.body.appendChild(el);
  pollUntilBack(el);
}

async function pollUntilBack(el) {
  // Brief pause so the server process has time to begin shutting down.
  await new Promise(r => setTimeout(r, 1500));
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch("/healthz");
      if (r.ok) {
        el.remove();
        refresh();
        return;
      }
    } catch (_) { /* server temporarily down — keep polling */ }
    await new Promise(r => setTimeout(r, 1000));
  }
  el.querySelector(".r-title").textContent = "⚠ Restart timed out";
  el.querySelector(".r-sub").textContent = "Please reload the page manually.";
}

async function dismissProviderWarning() {
  // Clear the persistent warning from the store so it won't reappear until
  // the next limit event. Uses a lightweight PATCH-style endpoint.
  try { await fetch("/api/dismiss-provider-warning", {method: "POST"}); } catch (_) {}
  const el = $("provider-warning-banner");
  if (el) el.classList.remove("show");
}

$("settings-btn").addEventListener("click", openSettings);
$("settings-cancel").addEventListener("click", () => $("settings-overlay").classList.remove("show"));
$("settings-save").addEventListener("click", saveSettings);
$("settings-overlay").addEventListener("click", (e) => {
  if (e.target.id === "settings-overlay") $("settings-overlay").classList.remove("show");
});

refresh();
setInterval(refresh, 5000);
