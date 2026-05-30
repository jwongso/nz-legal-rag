'use strict';

// ---- Debug mode (Ctrl+Shift+D to activate) ----
let _debugKey = '';
let _debugMode = false;

const compareGrid = document.getElementById('compare-grid');

function _getSelectedStrategies() {
  return [...document.querySelectorAll('.strategy-check:checked')].map(cb => cb.value);
}

function _setDebugUI(on) {
  document.getElementById('debug-badge').classList.toggle('hidden', !on);
  document.getElementById('debug-strategy').style.display = on ? 'flex' : '';
  if (!on) {
    compareGrid.style.display = '';
    compareGrid.innerHTML = '';
    askAnotherRow.classList.remove('visible');
  }
}

async function _activateDebug() {
  if (_debugMode) {
    _debugMode = false;
    _debugKey = '';
    _setDebugUI(false);
    history.replaceState(null, '', location.pathname);
    alert('Debug mode off.');
    return;
  }
  const key = prompt('Debug key:');
  if (!key) return;
  try {
    const r = await fetch('/debug/ping', {
      headers: { 'X-API-Key': _apiToken, 'X-Debug-Key': key },
    });
    if (r.ok) {
      _debugKey = key;
      _debugMode = true;
      _setDebugUI(true);
    } else {
      alert('Invalid debug key.');
    }
  } catch (_) {
    alert('Could not validate debug key.');
  }
}

function _initDebugShortcut() {
  // URL hash trigger: navigate to #debug to activate
  if (location.hash === '#debug') {
    history.replaceState(null, '', location.pathname);
    _activateDebug();
  }
  window.addEventListener('hashchange', () => {
    if (location.hash === '#debug') {
      history.replaceState(null, '', location.pathname);
      _activateDebug();
    }
  });

}

const _STRATEGY_LABELS = {
  vector: 'Vector',
  vector_no_legal: 'Vector (no legal rerank)',
  mmr: 'MMR diverse',
  bm25: 'BM25 keyword',
};

function _renderDebugPanel(dbg, dbgDone) {
  const existing = document.getElementById('debug-panel');
  if (existing) existing.remove();

  const scores = dbg.scores || [];
  const isBm25 = dbg.strategy === 'bm25';
  const maxScore = Math.max(...scores, 0.0001);

  const barData = scores.map((s, i) => {
    const pct = isBm25 ? Math.round((s / maxScore) * 100) : Math.round(s * 100);
    const cls = isBm25 ? 'mid' : (s >= 0.80 ? 'high' : s >= 0.76 ? 'mid' : 'low');
    const label = isBm25 ? s.toFixed(5) : s.toFixed(4);
    return { pct, cls, label, i };
  });
  const bars = barData.map(({ pct, cls, label, i }) =>
    `<div class="debug-score-row">
      <span class="debug-score-label">S${i + 1}</span>
      <div class="debug-score-bar-wrap"><div class="debug-score-bar ${cls}" data-pct="${pct}"></div></div>
      <span class="debug-score-val">${label}</span>
    </div>`
  ).join('');

  const strategyLabel = _STRATEGY_LABELS[dbg.strategy] || dbg.strategy || 'vector';
  const scoreNote = isBm25 ? ' <span class="debug-note">(BM25 scale, bars normalised)</span>' : '';

  const stats = `<div class="debug-stats">
    top <span>${dbg.top}</span> &nbsp;|&nbsp;
    min <span>${dbg.min}</span> &nbsp;|&nbsp;
    avg <span>${dbg.avg}</span> &nbsp;|&nbsp;
    chunks <span>${dbg.chunks}</span> &nbsp;|&nbsp;
    retrieve <span>${dbg.retrieve_ms}ms</span>
    ${dbgDone ? `&nbsp;|&nbsp; generate <span>${dbgDone.generate_ms}ms</span> &nbsp;|&nbsp; total <span>${dbgDone.total_ms}ms</span>` : ''}
  </div>`;

  const panel = document.createElement('div');
  panel.id = 'debug-panel';
  panel.className = 'debug-panel';
  panel.innerHTML = `<h4>Retrieval debug &mdash; <em>${strategyLabel}</em>${scoreNote}</h4>${bars}${stats}`;
  panel.querySelectorAll('.debug-score-bar[data-pct]').forEach(el => {
    el.style.width = el.dataset.pct + '%';
  });
  resultCard.appendChild(panel);
}

function _renderWebPanel(webEvent, container) {
  const existing = document.getElementById('web-panel');
  if (existing) existing.remove();

  const cached = webEvent.cached;
  const results = webEvent.results || [];

  const badge = cached
    ? '<span class="web-badge web-badge-cached">CACHED (7d)</span>'
    : '<span class="web-badge web-badge-live">LIVE</span>';

  const rows = results.map(r => {
    const domain = (() => { try { return new URL(r.url).hostname; } catch (_) { return r.url; } })();
    return `<div class="web-result-row">
      <a class="web-result-link" href="${escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title)}</a>
      <span class="web-result-domain">${escapeHtml(domain)}</span>
      <div class="web-result-body">${escapeHtml(r.body)}</div>
    </div>`;
  }).join('');

  const panel = document.createElement('div');
  panel.id = 'web-panel';
  panel.innerHTML = `<h4>Web verify ${badge}</h4>${rows || '<span class="debug-note">no results</span>'}`;
  (container || resultCard).appendChild(panel);
}

function _buildAnchorCard(s, anchorMethod) {
  const num = (s.document_id || '').replace('NZLEG/RTA/', '');
  const forbidden = s.forbidden_terms || {};
  const checks = _FORBIDDEN_ANCHOR_TERMS_DISPLAY.map(t => {
    const found = forbidden[t];
    return `<span class="${found ? 'ctx-forbidden-fail' : 'ctx-forbidden-ok'}">${escapeHtml(t)}: ${found ? 'YES' : 'no'}</span>`;
  }).join(' | ');
  const noText = !s.tokens || s.tokens === 0;
  const card = document.createElement('div');
  card.className = 'ctx-card ctx-card-leg';
  card.innerHTML = `<div class="ctx-card-header">${escapeHtml(num)} - ${escapeHtml(s.title || '')}</div>
<div class="ctx-card-meta">legislation | ${escapeHtml(anchorMethod || '')} | ~${s.tokens ?? '?'} tokens | score: n/a</div>
${noText ? '<div class="ctx-anchor-warn">Warning: anchor section selected but no text extracted - section was not sent to model. Heading pattern may not match or section lacks subsection markers.</div>' : ''}
<div class="ctx-card-forbidden">Forbidden terms: ${checks}</div>
<div class="ctx-card-preview">${escapeHtml((s.preview || '').slice(0, 400))}</div>`;
  return card;
}

function _buildChunkCard(c) {
  const scoreStr = c.score != null ? c.score.toFixed(4) : 'n/a';
  const gateMeta = c.passed_gate !== undefined ? ` | gate: ${c.passed_gate ? 'yes' : 'no'}` : '';
  const preview = (c.preview || '').slice(0, 300);
  const fullText = c.full_text || preview;
  const hasMore = fullText.length > preview.length;
  const card = document.createElement('div');
  card.className = 'ctx-card ctx-card-case';
  card.id = `ctx-S${c.source_index}`;
  card.innerHTML = `<div class="ctx-card-header">[S${c.source_index}] ${escapeHtml(c.document_id || '')}</div>
<div class="ctx-card-meta">case | score: ${escapeHtml(scoreStr)} | date: ${escapeHtml(c.date || '?')} | ~${c.tokens ?? '?'} tokens${escapeHtml(gateMeta)}</div>
<div class="ctx-card-preview">${escapeHtml(preview)}</div>
${hasMore ? `<button class="ctx-expand-btn">Show full chunk</button>` : ''}`;
  if (hasMore) {
    const btn = card.querySelector('.ctx-expand-btn');
    btn.dataset.full = fullText;
    btn.dataset.preview = preview;
  }
  return card;
}

// ---- Delegated: citation link -> scroll + highlight matching context card ----
document.addEventListener('click', e => {
  const link = e.target.closest('.citation-link');
  if (!link) return;
  e.preventDefault();
  const src = link.dataset.source;
  // Scope search to the same column in compare mode, fall back to whole document.
  const scope = link.closest('.compare-col') || document;
  const card = scope.querySelector(`#ctx-${CSS.escape(src)}`);
  if (!card) return;
  // Open parent <details> if collapsed.
  const det = card.closest('details');
  if (det && !det.open) det.open = true;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  card.classList.remove('citation-highlight');
  void card.offsetWidth; // force reflow to restart transition
  card.classList.add('citation-highlight');
  setTimeout(() => card.classList.remove('citation-highlight'), 2500);
});

// ---- Delegated: expand/collapse full chunk text ----
document.addEventListener('click', e => {
  const btn = e.target.closest('.ctx-expand-btn');
  if (!btn) return;
  const preview = btn.previousElementSibling;
  if (!preview) return;
  const expanded = btn.dataset.expanded === 'true';
  preview.textContent = expanded ? btn.dataset.preview : btn.dataset.full;
  btn.dataset.expanded = expanded ? 'false' : 'true';
  btn.textContent = expanded ? 'Show full chunk' : 'Collapse';
});

function _buildBudgetMeter(budget) {
  if (!budget) return null;
  const pct = Math.min(100, Math.round((budget.total_tokens / budget.ctx_limit) * 100));
  const cls = pct >= 80 ? 'ctx-budget-high' : pct >= 50 ? 'ctx-budget-mid' : 'ctx-budget-low';
  const truncNote = budget.truncated_chunks > 0 ? ` | truncated: ${budget.truncated_chunks}` : '';
  const wrap = document.createElement('div');
  wrap.className = 'ctx-budget';
  wrap.innerHTML = `<div class="ctx-budget-bar-wrap"><div class="ctx-budget-bar ${cls}" data-pct="${pct}"></div></div>
<div class="ctx-budget-label">Context: ${budget.total_tokens.toLocaleString()} / ${budget.ctx_limit.toLocaleString()} tokens (~${pct}%)</div>
<div class="ctx-budget-detail">Anchor: ~${budget.anchor_tokens} tk | Chunks: ~${budget.chunk_tokens} tk | Sources: ${budget.sources_sent}${escapeHtml(truncNote)}</div>`;
  const bar = wrap.querySelector('.ctx-budget-bar[data-pct]');
  if (bar) bar.style.width = pct + '%';
  return wrap;
}

function _renderContextDebugPanel(ev, container) {
  const existing = (container || resultCard).querySelector('.context-debug-panel');
  if (existing) existing.remove();

  const panel = document.createElement('details');
  panel.className = 'context-debug-panel';

  const summary = document.createElement('summary');
  summary.className = 'context-debug-toggle';
  summary.textContent = 'Context sent to model';
  panel.appendChild(summary);

  const body = document.createElement('div');
  body.className = 'context-debug-body';

  // Budget meter
  const budgetEl = _buildBudgetMeter(ev.budget);
  if (budgetEl) body.appendChild(budgetEl);

  // Query + planner block
  const pl = ev.planner || {};
  const rewriteChanged = ev.rewritten_query && ev.rewritten_query !== ev.original_query;
  let qHtml = '<div class="ctx-query-block">';
  qHtml += `<div class="ctx-query-row"><span class="ctx-label">Original query</span><span class="ctx-query-text">${escapeHtml(ev.original_query || '')}</span></div>`;
  if (ev.rewritten_query !== undefined) {
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Rewritten query</span><span class="ctx-query-text${rewriteChanged ? ' ctx-rewrite-changed' : ''}">${escapeHtml(ev.rewritten_query || ev.original_query || '')}</span></div>`;
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Rewrite</span><span class="ctx-meta-val">${ev.rewrite_used ? 'yes' : 'no'}</span></div>`;
  }
  if (pl.property_change_triggered) {
    const sections = (pl.forced_sections || []).map(s => s.replace('NZLEG/RTA/', '')).join(', ');
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Prop-change gate</span><span class="ctx-meta-val ctx-gate-yes">triggered | terms: ${escapeHtml((pl.trigger_terms || []).join(', '))} | forced: ${escapeHtml(sections)}</span></div>`;
    const gate = pl.gate || {};
    const fallbackNote = gate.fallback_used ? ' | FALLBACK: all filtered, using original' : '';
    const rejectedNote = gate.rejected && gate.rejected.length ? ` | rejected: ${escapeHtml(gate.rejected.join(', '))}` : '';
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Gate filter</span><span class="ctx-meta-val">before: ${gate.candidates_before ?? '?'} | survived: ${gate.survived ?? '?'}${escapeHtml(fallbackNote)}${rejectedNote}</span></div>`;
  }
  qHtml += '</div>';
  const qBlock = document.createElement('div');
  qBlock.innerHTML = qHtml;
  body.appendChild(qBlock.firstElementChild);

  // Anchor cards
  const anchor = ev.anchor || {};
  if (anchor.sections && anchor.sections.length) {
    const lbl = document.createElement('div');
    lbl.className = 'ctx-section-label';
    lbl.textContent = `RTA anchor - ${anchor.method || 'unknown'} (legislation, not [SN] cited)`;
    body.appendChild(lbl);
    anchor.sections.forEach(s => body.appendChild(_buildAnchorCard(s, anchor.method)));
  }

  // Chunk cards
  const chunks = ev.chunks || [];
  if (chunks.length) {
    const lbl = document.createElement('div');
    lbl.className = 'ctx-section-label';
    lbl.textContent = `Case chunks (${chunks.length}) - click [SN] in answer to jump here`;
    body.appendChild(lbl);
    chunks.forEach(c => body.appendChild(_buildChunkCard(c)));
  }

  panel.appendChild(body);
  (container || resultCard).appendChild(panel);
}

function _renderSharedContextDebugPanel(ev, container) {
  const existing = container.querySelector('.shared-context-debug-panel');
  if (existing) existing.remove();

  const panel = document.createElement('details');
  panel.className = 'context-debug-panel shared-context-debug-panel';

  const summary = document.createElement('summary');
  summary.className = 'context-debug-toggle';
  summary.textContent = 'Shared context (all strategies)';
  panel.appendChild(summary);

  const body = document.createElement('div');
  body.className = 'context-debug-body';

  const pl = ev.planner || {};
  let qHtml = '<div class="ctx-query-block">';
  qHtml += `<div class="ctx-query-row"><span class="ctx-label">Original query</span><span class="ctx-query-text">${escapeHtml(ev.original_query || '')}</span></div>`;
  qHtml += `<div class="ctx-query-row"><span class="ctx-label">Rewrite</span><span class="ctx-meta-val">disabled in compare mode - all strategies use the raw query</span></div>`;
  if (pl.property_change_triggered) {
    const sections = (pl.forced_sections || []).map(s => s.replace('NZLEG/RTA/', '')).join(', ');
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Prop-change gate</span><span class="ctx-meta-val ctx-gate-yes">triggered | terms: ${escapeHtml((pl.trigger_terms || []).join(', '))} | forced: ${escapeHtml(sections)}</span></div>`;
  }
  qHtml += '</div>';
  const qBlock = document.createElement('div');
  qBlock.innerHTML = qHtml;
  body.appendChild(qBlock.firstElementChild);

  const anchor = ev.anchor || {};
  if (anchor.sections && anchor.sections.length) {
    const lbl = document.createElement('div');
    lbl.className = 'ctx-section-label';
    lbl.textContent = `RTA anchor - ${anchor.method || 'unknown'} (shared, not [SN] cited)`;
    body.appendChild(lbl);
    anchor.sections.forEach(s => body.appendChild(_buildAnchorCard(s, anchor.method)));
  }

  panel.appendChild(body);
  container.appendChild(panel);
}

let _apiToken = '';

async function _loadToken() {
  try {
    const res = await fetch('/token');
    const data = await res.json();
    _apiToken = data.token || '';
  } catch (_) {}
}

const form = document.getElementById('ask-form');
const questionEl = document.getElementById('question');
const charCountEl = document.getElementById('char-count');
const submitBtn = document.getElementById('submit-btn');
const queueNotice = document.getElementById('queue-notice');
const loadingCard = document.getElementById('loading-card');
const loadingText = document.getElementById('loading-text');
const resultCard = document.getElementById('result-card');
const answerBody = document.getElementById('answer-body');
const sourcesSection = document.getElementById('sources-section');
const sourcesList = document.getElementById('sources-list');
const errorCard = document.getElementById('error-card');
const errorText = document.getElementById('error-text');
const askAnotherRow = document.getElementById('ask-another-row');

// Feedback elements
const thumbUp = document.getElementById('thumb-up');
const thumbDown = document.getElementById('thumb-down');
const feedbackComment = document.getElementById('feedback-comment');
const feedbackText = document.getElementById('feedback-text');
const feedbackSubmit = document.getElementById('feedback-submit');
const feedbackThanks = document.getElementById('feedback-thanks');

const app_js_version = 30;
const LOADING_MESSAGES = [
  'Searching through Tenancy Tribunal decisions...',
  'Analysing relevant cases...',
  'Preparing your answer...',
  'Almost there...',
];

let loadingInterval = null;
let loadingStep = 0;
let currentQuestion = '';
let currentRating = null;
let _debugInfo = null;
let _webResultsInfo = null;
let _sharedContextDebugInfo = null;
let _artifact = {};
let _colArtifacts = {};

// Terms shown in anchor forbidden-term checklist (must match backend _FORBIDDEN_ANCHOR_TERMS).
const _FORBIDDEN_ANCHOR_TERMS_DISPLAY = [
  'Schedule 1A', 'infringement fee', '42A(7)', '19(2)', 'penalty notice',
];

// ---- Character counter ----
questionEl.addEventListener('input', () => {
  const len = questionEl.value.length;
  charCountEl.textContent = len;
  charCountEl.parentElement.classList.toggle('near-limit', len > 4500);
});

// ---- Example question buttons ----
document.querySelectorAll('.example-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    questionEl.value = btn.dataset.q;
    questionEl.dispatchEvent(new Event('input'));
    questionEl.focus();
  });
});

// ---- Queue status polling ----
async function pollQueue() {
  try {
    const res = await fetch('/health');
    if (!res.ok) return;
    const data = await res.json();
    const waiting = data.waiting || 0;
    const est = data.estimated_wait_seconds || 0;
    if (waiting > 0) {
      queueNotice.textContent = `${waiting} ${waiting === 1 ? 'person' : 'people'} waiting - estimated wait ~${est}s`;
      queueNotice.classList.add('visible');
    } else {
      queueNotice.classList.remove('visible');
    }
  } catch (_) { /* ignore */ }
}

// ---- Answer rendering ----
function renderAnswer(text) {
  const idx = text.lastIndexOf('\n\nSources:');
  if (idx !== -1) text = text.substring(0, idx);
  text = escapeHtml(text.trim());

  // Build structure first (before inline replacements so ** and [SN] don't
  // interfere with list detection - e.g. **3. Item** would break ^\d+\. )
  const html = text.split(/\n{2,}/).map(para => {
    const lines = para.split('\n');

    if (lines.some(l => /^[-*] /.test(l.trim()))) {
      // Accumulate continuation lines into the current bullet item
      const items = [];
      let cur = null;
      for (const line of lines) {
        if (/^[-*] /.test(line.trim())) {
          if (cur !== null) items.push(cur);
          cur = line.trim().replace(/^[-*] /, '').replace(/  $/, '');
        } else if (cur !== null && line.trim()) {
          cur += ' ' + line.trim();
        }
      }
      if (cur !== null) items.push(cur);
      return `<ul>${items.map(t => `<li>${t}</li>`).join('')}</ul>`;
    }

    if (lines.some(l => /^\d+\. /.test(l.trim()))) {
      // Accumulate continuation lines into the current numbered item
      const items = [];
      let cur = null;
      for (const line of lines) {
        const m = line.trim().match(/^(\d+)\. (.*)/);
        if (m) {
          if (cur) items.push(cur);
          cur = { num: m[1], text: m[2].replace(/  $/, '') };
        } else if (cur && line.trim()) {
          cur.text += ' ' + line.trim();
        }
      }
      if (cur) items.push(cur);
      return `<ol>${items.map(it => `<li value="${it.num}">${it.text}</li>`).join('')}</ol>`;
    }

    return `<p>${lines.map(l => l.replace(/  $/, '')).join('<br>')}</p>`;
  }).join('');

  // Apply inline formatting after structure is built
  return html
    .replace(/\[S(\d+)\]/g, '<a href="#ctx-S$1" class="citation-link" data-source="S$1">[S$1]</a>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderSources(sources, legislation) {
  const hasLeg = legislation && legislation.length > 0;
  const hasDec = sources && sources.length > 0;
  if (!hasLeg && !hasDec) {
    sourcesSection.classList.remove('visible');
    return;
  }
  let html = '';
  if (hasLeg) {
    if (hasDec) html += '<div class="sources-group-label">Relevant legislation</div>';
    html += legislation.map(s => {
      const url = (s.url || '').startsWith('https://') ? s.url : '#';
      const secMatch = (s.case_id || '').match(/\/s(\d+[A-Z]?)$/i);
      const secNum = secMatch ? secMatch[1] : '';
      const dataAttr = secNum ? ` data-section="${escapeHtml(secNum)}"` : '';
      return `
        <div class="source-card source-card--leg">
          <span class="source-num source-num--leg leg-sec-toggle"${dataAttr} title="Show decisions citing this section">&sect;</span>
          <div class="source-info">
            <a class="source-title" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.title || s.case_id)}</a>
          </div>
        </div>`;
    }).join('');
  }
  if (hasDec) {
    if (hasLeg) html += '<div class="sources-group-label">Tribunal decisions</div>';
    html += sources.map((s, i) => {
      const court = s.court_name || 'Tenancy Tribunal';
      const date = s.date || '';
      const label = date ? `${court} Decision - ${date}` : `${court} Decision`;
      const url = (s.url || '').startsWith('https://') ? s.url : '#';
      return `
        <div class="source-card">
          <span class="source-num">S${i + 1}</span>
          <div class="source-info">
            <a class="source-title" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>
          </div>
        </div>`;
    }).join('');
  }
  sourcesList.innerHTML = html;
  sourcesSection.classList.add('visible');
}

function renderConfidence(ev) {
  const existing = document.getElementById('confidence-badge');
  if (existing) existing.remove();
  if (!ev || !ev.level) return;
  const badge = document.createElement('div');
  badge.id = 'confidence-badge';
  badge.className = `confidence-badge confidence-${ev.level}`;
  const icons = { high: '●', medium: '◑', low: '○' };
  badge.innerHTML = `<span class="confidence-icon">${icons[ev.level] || '●'}</span> <span class="confidence-msg">${escapeHtml(ev.message)}</span>`;
  const aiWarning = resultCard.querySelector('.ai-warning');
  if (aiWarning) resultCard.insertBefore(badge, aiWarning);
}

function renderVerification(sections) {
  const existing = document.getElementById('verification-panel');
  if (existing) existing.remove();
  if (!sections || !sections.length) return;
  const panel = document.createElement('div');
  panel.id = 'verification-panel';
  panel.className = 'verification-panel';
  let html = '<div class="verification-header"><span class="verification-check">&#10003;</span> Verified against current legislation (legislation.govt.nz)</div>';
  sections.forEach(s => {
    html += `<details class="verification-item">
      <summary class="verification-ref"><a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.reference)}</a> - click to expand</summary>
      <pre class="verification-excerpt">${escapeHtml(s.excerpt)}</pre>
    </details>`;
  });
  panel.innerHTML = html;
  resultCard.appendChild(panel);
}

async function fetchLegislationCases(sectionNum, badgeEl) {
  const existing = badgeEl.closest('.source-card--leg').querySelector('.leg-cases-panel');
  if (existing) { existing.remove(); return; }
  const panel = document.createElement('div');
  panel.className = 'leg-cases-panel';
  panel.textContent = 'Loading...';
  badgeEl.closest('.source-card--leg').appendChild(panel);
  try {
    const r = await fetch(`/legislation/cases?section=${encodeURIComponent(sectionNum)}&limit=8`, {
      headers: { 'X-API-Key': _apiToken },
    });
    const data = await r.json();
    const cases = data.cases || [];
    if (!cases.length) {
      panel.textContent = 'No indexed decisions found for this section.';
      return;
    }
    panel.innerHTML = `<div class="leg-cases-label">Decisions citing s${escapeHtml(sectionNum)}</div>` +
      cases.map(c => {
        const url = (c.url || '').startsWith('https://') ? c.url : '#';
        const date = c.date || '';
        const n = c.mentions || 1;
        return `<div class="leg-case-row"><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(date)}</a><span class="leg-case-mentions">${n}x</span></div>`;
      }).join('');
  } catch (_) {
    panel.textContent = 'Could not load decisions.';
  }
}

// ---- Artifact accumulator ----
function _resetArtifact(question, strategy) {
  _artifact = {
    question,
    strategy,
    irac: document.getElementById('irac-toggle').checked,
    think: _debugMode && document.getElementById('think-toggle').checked,
    debug_mode: _debugMode,
    ts_start: new Date().toISOString(),
    ts_end: null,
    user_agent: navigator.userAgent,
    viewport: { w: window.innerWidth, h: window.innerHeight },
    answer: '',
    sources: [],
    legislation: [],
    confidence: null,
    web_results: null,
    verification: null,
    debug: null,
    debug_timing: null,
    context_debug: null,
  };
}

async function _saveFullFeedback(payload, rating, comment) {
  try {
    await fetch('/feedback/full', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': _apiToken },
      body: JSON.stringify({ ...payload, rating, comment: comment || '' }),
    });
  } catch (_) {}
}

// ---- Feedback ----
function resetFeedback() {
  currentRating = null;
  thumbUp.classList.remove('active');
  thumbDown.classList.remove('active');
  feedbackComment.style.display = 'none';
  feedbackText.value = '';
  feedbackThanks.style.display = 'none';
  document.getElementById('feedback-row').style.display = 'flex';
}

function submitFeedback(rating) {
  if (currentRating === rating) {
    currentRating = null;
    thumbUp.classList.remove('active');
    thumbDown.classList.remove('active');
    feedbackComment.style.display = 'none';
    return;
  }
  currentRating = rating;
  thumbUp.classList.toggle('active', rating === 1);
  thumbDown.classList.toggle('active', rating === -1);
  feedbackComment.style.display = 'block';
  if (rating === -1) {
    _saveFullFeedback(_artifact, -1, '');
  }
}

thumbUp.addEventListener('click', () => submitFeedback(1));
thumbDown.addEventListener('click', () => submitFeedback(-1));

feedbackSubmit.addEventListener('click', async () => {
  if (currentRating === null) return;
  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': _apiToken },
      body: JSON.stringify({
        question: currentQuestion,
        rating: currentRating,
        comment: feedbackText.value.trim(),
      }),
    });
  } catch (_) { /* ignore network errors on feedback */ }
  feedbackComment.style.display = 'none';
  document.getElementById('feedback-row').style.display = 'none';
  feedbackThanks.style.display = 'block';
});

// ---- State helpers ----
function showLoading() {
  const vp = document.getElementById('verification-panel');
  if (vp) vp.remove();
  loadingCard.classList.add('visible');
  resultCard.classList.remove('visible');
  errorCard.classList.remove('visible');
  sourcesSection.classList.remove('visible');
  askAnotherRow.classList.remove('visible');
  submitBtn.disabled = true;
  loadingStep = 0;
  loadingText.textContent = LOADING_MESSAGES[0];
  loadingInterval = setInterval(() => {
    loadingStep = (loadingStep + 1) % LOADING_MESSAGES.length;
    loadingText.textContent = LOADING_MESSAGES[loadingStep];
  }, 5000);
}

function showStreamingResult() {
  clearInterval(loadingInterval);
  loadingCard.classList.remove('visible');
  errorCard.classList.remove('visible');
  resultCard.classList.add('visible');
  askAnotherRow.classList.add('visible');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function finaliseResult(fullText, sources, legislation) {
  _artifact.answer = fullText;
  _artifact.ts_end = new Date().toISOString();
  answerBody.innerHTML = renderAnswer(fullText);
  renderSources(sources, legislation);
  resetFeedback();
  submitBtn.disabled = false;
}

function showError(message) {
  clearInterval(loadingInterval);
  loadingCard.classList.remove('visible');
  resultCard.classList.remove('visible');
  sourcesSection.classList.remove('visible');
  errorText.textContent = message;
  errorCard.classList.add('visible');
  submitBtn.disabled = false;
}

function resetToForm() {
  resultCard.classList.remove('visible');
  errorCard.classList.remove('visible');
  sourcesSection.classList.remove('visible');
  askAnotherRow.classList.remove('visible');
  const cb = document.getElementById('confidence-badge');
  if (cb) cb.remove();
  const vp = document.getElementById('verification-panel');
  if (vp) vp.remove();
  const wp = document.getElementById('web-panel');
  if (wp) wp.remove();
  const dp = document.getElementById('debug-panel');
  if (dp) dp.remove();
  resultCard.querySelectorAll('.context-debug-panel').forEach(p => p.remove());
  compareGrid.querySelectorAll('.context-debug-panel').forEach(p => p.remove());
  questionEl.value = '';
  charCountEl.textContent = '0';
  questionEl.focus();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---- Compare mode helpers ----
function _colId(strategy) { return 'col-' + strategy.replace(/_/g, '-'); }

function _buildCompareColumns(strategies) {
  compareGrid.innerHTML = '';
  compareGrid.style.setProperty('--col-count', strategies.length);
  compareGrid.style.display = 'grid';
  loadingCard.classList.remove('visible');
  resultCard.classList.remove('visible');
  sourcesSection.classList.remove('visible');
  errorCard.classList.remove('visible');
  askAnotherRow.classList.remove('visible');

  strategies.forEach(strat => {
    const col = document.createElement('div');
    col.className = 'compare-col';
    col.id = _colId(strat);
    col.innerHTML = `
      <div class="compare-col-header">${_STRATEGY_LABELS[strat] || strat}</div>
      <div class="compare-col-scores"></div>
      <div class="compare-col-body">
        <div class="compare-col-spinner"><div class="spinner-sm"></div> Waiting...</div>
        <div class="compare-col-answer" style="display:none"></div>
      </div>
      <div class="compare-col-sources"></div>`;
    compareGrid.appendChild(col);
  });

  window.scrollTo({ top: compareGrid.offsetTop - 20, behavior: 'smooth' });
}

function _colSetActive(strategy) {
  const col = document.getElementById(_colId(strategy));
  if (!col) return;
  col.querySelector('.compare-col-spinner').style.display = 'flex';
  col.querySelector('.compare-col-answer').style.display = 'none';
  col.querySelector('.compare-col-answer').textContent = '';
}

function _colAppendToken(strategy, text) {
  const col = document.getElementById(_colId(strategy));
  if (!col) return;
  const spinner = col.querySelector('.compare-col-spinner');
  const answer = col.querySelector('.compare-col-answer');
  if (spinner.style.display !== 'none') {
    spinner.style.display = 'none';
    answer.style.display = 'block';
  }
  answer.textContent += text;
}

function _colFinalise(strategy) {
  const col = document.getElementById(_colId(strategy));
  if (!col) return;
  const answer = col.querySelector('.compare-col-answer');
  answer.innerHTML = renderAnswer(answer.textContent);
  _colAddFeedback(col, strategy);
}

function _colAddFeedback(col, strategy) {
  const fb = document.createElement('div');
  fb.className = 'col-feedback';
  col.appendChild(fb);

  // Build elements explicitly so initial visibility is unambiguous
  const row = document.createElement('div');
  row.className = 'col-feedback-row';
  row.innerHTML = `<span class="col-feedback-label">Helpful?</span>`;
  const upBtn = document.createElement('button');
  upBtn.className = 'col-thumb col-thumb-up';
  upBtn.title = 'Yes';
  upBtn.textContent = '👍';
  const downBtn = document.createElement('button');
  downBtn.className = 'col-thumb col-thumb-down';
  downBtn.title = 'No';
  downBtn.textContent = '👎';
  row.appendChild(upBtn);
  row.appendChild(downBtn);
  fb.appendChild(row);

  const commentBox = document.createElement('div');
  commentBox.className = 'col-feedback-comment';
  commentBox.style.display = 'none';
  const textarea = document.createElement('textarea');
  textarea.className = 'col-feedback-text';
  textarea.rows = 2;
  textarea.placeholder = 'What could be better? (optional)';
  const sendBtn = document.createElement('button');
  sendBtn.className = 'col-feedback-submit btn-secondary';
  sendBtn.textContent = 'Send';
  commentBox.appendChild(textarea);
  commentBox.appendChild(sendBtn);
  fb.appendChild(commentBox);

  const thanks = document.createElement('div');
  thanks.className = 'col-feedback-thanks';
  thanks.style.display = 'none';
  thanks.textContent = 'Thanks!';
  fb.appendChild(thanks);

  let rating = null;

  function selectRating(r) {
    if (rating === r) {
      rating = null;
      upBtn.classList.remove('active');
      downBtn.classList.remove('active');
      commentBox.style.display = 'none';
      return;
    }
    rating = r;
    upBtn.classList.toggle('active', r === 1);
    downBtn.classList.toggle('active', r === -1);
    commentBox.style.display = 'block';
  }

  upBtn.addEventListener('click', () => selectRating(1));
  downBtn.addEventListener('click', () => {
    selectRating(-1);
    _saveFullFeedback(_colArtifacts[strategy] || { question: currentQuestion, strategy }, -1, '');
  });

  sendBtn.addEventListener('click', async () => {
    if (rating === null) return;
    const comment = textarea.value.trim();
    try {
      await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': _apiToken },
        body: JSON.stringify({
          question: currentQuestion,
          rating,
          comment: `[${strategy}] ${comment}`.trim(),
        }),
      });
    } catch (_) {}
    row.style.display = 'none';
    commentBox.style.display = 'none';
    thanks.style.display = 'block';
  });
}

function _colSetThink(strategy, text) {
  const col = document.getElementById(_colId(strategy));
  if (!col) return;
  const existing = col.querySelector('.compare-col-think');
  if (existing) return;
  const details = document.createElement('details');
  details.className = 'compare-col-think';
  details.innerHTML = `<summary>Reasoning <span class="think-len">${text.length} chars</span></summary><pre>${escapeHtml(text)}</pre>`;
  col.querySelector('.compare-col-body').insertBefore(details, col.querySelector('.compare-col-answer'));
}

function _colSetSources(strategy, sources, legislation) {
  const col = document.getElementById(_colId(strategy));
  if (!col) return;
  const hasLeg = legislation && legislation.length > 0;
  const hasDec = sources && sources.length > 0;
  if (!hasLeg && !hasDec) return;
  let html = '<div class="compare-sources-label">Sources</div>';
  if (hasLeg) {
    if (hasDec) html += '<div class="compare-sources-group">&sect; Legislation</div>';
    html += legislation.map(s => {
      const url = (s.url || '').startsWith('https://') ? s.url : '#';
      return `<div class="compare-source-row"><span class="source-num source-num--leg">&sect;</span> <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.title || s.case_id)}</a></div>`;
    }).join('');
  }
  if (hasDec) {
    if (hasLeg) html += '<div class="compare-sources-group">Decisions</div>';
    html += sources.map((s, i) => {
      const label = s.date ? `${s.court_name || 'Tribunal'} - ${s.date}` : (s.court_name || 'Tribunal');
      const url = (s.url || '').startsWith('https://') ? s.url : '#';
      return `<div class="compare-source-row"><span class="source-num">S${i+1}</span> <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a></div>`;
    }).join('');
  }
  col.querySelector('.compare-col-sources').innerHTML = html;
}

function _colSetScores(strategy, dbg) {
  const col = document.getElementById(_colId(strategy));
  if (!col) return;
  const isBm25 = strategy === 'bm25';
  const scores = dbg.scores || [];
  const maxScore = Math.max(...scores, 0.0001);
  col.querySelector('.compare-col-scores').innerHTML =
    scores.map((s, i) => {
      const pct = isBm25 ? Math.round((s / maxScore) * 100) : Math.round(s * 100);
      const cls = isBm25 ? 'mid' : (s >= 0.80 ? 'high' : s >= 0.76 ? 'mid' : 'low');
      return `<div class="compare-score-row"><span class="compare-score-label">S${i+1}</span><div class="compare-score-bar-wrap"><div class="debug-score-bar ${cls}" style="width:${pct}%"></div></div><span class="compare-score-val">${isBm25 ? s.toFixed(5) : s.toFixed(4)}</span></div>`;
    }).join('') + `<div class="compare-score-stat">${dbg.retrieve_ms}ms retrieve</div>`;

  if (dbg.chunk_cards && dbg.chunk_cards.length) {
    const existing = col.querySelector('.context-debug-panel');
    if (existing) existing.remove();
    const miniPanel = document.createElement('details');
    miniPanel.className = 'context-debug-panel';
    const sum = document.createElement('summary');
    sum.className = 'context-debug-toggle';
    sum.textContent = `Case chunks (${dbg.chunk_cards.length})`;
    miniPanel.appendChild(sum);
    const body = document.createElement('div');
    body.className = 'context-debug-body';
    const lbl = document.createElement('div');
    lbl.className = 'ctx-section-label';
    lbl.textContent = '[SN] matches prompt';
    body.appendChild(lbl);
    dbg.chunk_cards.forEach(c => body.appendChild(_buildChunkCard(c)));
    miniPanel.appendChild(body);
    col.querySelector('.compare-col-scores').after(miniPanel);
  }
}

function _colSetError(strategy, msg) {
  const col = document.getElementById(_colId(strategy));
  if (!col) return;
  col.querySelector('.compare-col-spinner').style.display = 'none';
  const answer = col.querySelector('.compare-col-answer');
  answer.style.display = 'block';
  answer.innerHTML = `<span class="compare-error">${escapeHtml(msg)}</span>`;
}

async function _submitCompare(question, strategies) {
  submitBtn.disabled = true;
  _debugInfo = null;
  _webResultsInfo = null;
  _sharedContextDebugInfo = null;
  _colArtifacts = {};
  _buildCompareColumns(strategies);
  askAnotherRow.classList.remove('visible');

  let res;
  try {
    res = await fetch('/ask/stream/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': _apiToken },
      body: JSON.stringify({
        question,
        debug_key: _debugKey,
        strategies,
        thinking: document.getElementById('think-toggle').checked,
      }),
    });
  } catch (_) {
    compareGrid.style.display = 'none';
    showError('Could not connect to the server.');
    submitBtn.disabled = false;
    return;
  }

  if (!res.ok) {
    compareGrid.style.display = 'none';
    showError('Compare request failed.');
    submitBtn.disabled = false;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const colAnswers = {};

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        if (!raw.startsWith('data: ')) continue;
        let ev;
        try { ev = JSON.parse(raw.slice(6)); } catch (_) { continue; }

        const s = ev.strategy;
        if (ev.type === 'col_start') {
          _colSetActive(s);
          colAnswers[s] = '';
          _colArtifacts[s] = {
            question,
            strategy: s,
            irac: document.getElementById('irac-toggle').checked,
            think: document.getElementById('think-toggle').checked,
            debug_mode: true,
            ts_start: new Date().toISOString(),
            ts_end: null,
            user_agent: navigator.userAgent,
            viewport: { w: window.innerWidth, h: window.innerHeight },
            answer: '',
            sources: [],
            legislation: [],
            confidence: null,
            web_results: null,
            verification: null,
            debug: null,
            debug_timing: null,
            context_debug: null,
          };
        } else if (ev.type === 'col_sources') {
          _colSetSources(s, ev.sources, ev.legislation);
          if (_colArtifacts[s]) {
            _colArtifacts[s].sources = ev.sources || [];
            _colArtifacts[s].legislation = ev.legislation || [];
          }
        } else if (ev.type === 'col_debug') {
          _colSetScores(s, ev);
          if (_colArtifacts[s]) _colArtifacts[s].debug = ev;
        } else if (ev.type === 'col_think') {
          _colSetThink(s, ev.text);
        } else if (ev.type === 'col_token') {
          colAnswers[s] = (colAnswers[s] || '') + ev.text;
          if (_colArtifacts[s]) _colArtifacts[s].answer = colAnswers[s];
          _colAppendToken(s, ev.text);
        } else if (ev.type === 'col_done') {
          if (_colArtifacts[s]) {
            _colArtifacts[s].ts_end = new Date().toISOString();
            _colArtifacts[s].debug_timing = { generate_ms: ev.generate_ms, total_ms: ev.total_ms };
          }
          _colFinalise(s);
        } else if (ev.type === 'col_error') {
          _colSetError(s, ev.message);
        } else if (ev.type === 'shared_context_debug') {
          _sharedContextDebugInfo = ev;
          Object.values(_colArtifacts).forEach(a => { a.context_debug = ev; });
        } else if (ev.type === 'web_results') {
          _webResultsInfo = ev;
          Object.values(_colArtifacts).forEach(a => { a.web_results = ev; });
        } else if (ev.type === 'all_done') {
          if (_webResultsInfo) _renderWebPanel(_webResultsInfo, compareGrid);
          if (_sharedContextDebugInfo) _renderSharedContextDebugPanel(_sharedContextDebugInfo, compareGrid);
          askAnotherRow.classList.add('visible');
        }
      }
    }
  } catch (_) {
    // stream ended - finalise whatever we have
    Object.keys(colAnswers).forEach(s => _colFinalise(s));
  }
  submitBtn.disabled = false;
}

// ---- Form submit (SSE streaming) ----
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question) { questionEl.focus(); return; }
  currentQuestion = question;

  const strategies = _debugMode ? _getSelectedStrategies() : ['vector'];
  if (strategies.length === 0) {
    showError('Select at least one strategy.');
    return;
  }

  if (_debugMode && strategies.length > 1) {
    await _submitCompare(question, strategies);
    return;
  }

  showLoading();
  _debugInfo = null;
  _webResultsInfo = null;
  _sharedContextDebugInfo = null;
  _resetArtifact(question, strategies[0] || 'vector');

  let res;
  try {
    res = await fetch('/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': _apiToken },
      body: JSON.stringify({
        question,
        debug_key: _debugKey,
        strategy: strategies[0] || 'vector',
        irac: document.getElementById('irac-toggle').checked,
      }),
    });
  } catch (_) {
    showError('Could not connect to the server. Please check your connection and try again.');
    return;
  }

  if (!res.ok) {
    let msg = 'An error occurred.';
    try {
      const data = await res.json();
      msg = (data.detail && data.detail.error) || data.detail || msg;
    } catch (_) {}
    if (res.status === 429) {
      showError('You already have a query in progress. Please wait for it to finish.');
    } else if (res.status === 503) {
      showError('The server is busy right now. Please try again in a moment.');
    } else {
      showError(msg);
    }
    return;
  }

  // SSE: parse the stream
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let rawAnswer = '';
  let streamedSources = [];
  let streamedLegislation = [];
  let streamingStarted = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        if (!raw.startsWith('data: ')) continue;
        let event;
        try { event = JSON.parse(raw.slice(6)); } catch (_) { continue; }

        if (event.type === 'sources') {
          streamedSources = event.sources;
          streamedLegislation = event.legislation || [];
          _artifact.sources = event.sources || [];
          _artifact.legislation = event.legislation || [];
          renderSources(streamedSources, streamedLegislation);
        } else if (event.type === 'confidence') {
          _artifact.confidence = { level: event.level, message: event.message };
          renderConfidence(event);
        } else if (event.type === 'web_results') {
          _artifact.web_results = event;
          _webResultsInfo = event;
        } else if (event.type === 'debug') {
          _artifact.debug = event;
          _debugInfo = event;
        } else if (event.type === 'debug_done') {
          _artifact.debug_timing = { generate_ms: event.generate_ms, total_ms: event.total_ms };
          if (_debugInfo) _renderDebugPanel(_debugInfo, event);
        } else if (event.type === 'context_debug') {
          _artifact.context_debug = event;
          _renderContextDebugPanel(event, resultCard);
        } else if (event.type === 'token') {
          if (!streamingStarted) {
            streamingStarted = true;
            showStreamingResult();
            answerBody.textContent = '';
          }
          rawAnswer += event.text;
          _artifact.answer = rawAnswer;
          answerBody.textContent = rawAnswer;
        } else if (event.type === 'done') {
          finaliseResult(rawAnswer, streamedSources, streamedLegislation);
          if (_webResultsInfo) _renderWebPanel(_webResultsInfo);
        } else if (event.type === 'verification') {
          _artifact.verification = event.sections || [];
          renderVerification(event.sections);
        } else if (event.type === 'error') {
          showError(event.message || 'An error occurred.');
          return;
        }
      }
    }
    // If stream ended without a 'done' event, finalise what we have
    if (streamingStarted && rawAnswer) {
      finaliseResult(rawAnswer, streamedSources, streamedLegislation);
    }
  } catch (_) {
    showError('Lost connection while receiving the answer. Please try again.');
  }
});

// ---- Legislation section click (delegated) ----
sourcesList.addEventListener('click', e => {
  const toggle = e.target.closest('.leg-sec-toggle');
  if (!toggle || !toggle.dataset.section) return;
  fetchLegislationCases(toggle.dataset.section, toggle);
});

// ---- Ask another / retry ----
document.getElementById('ask-another-btn').addEventListener('click', resetToForm);
document.getElementById('retry-btn').addEventListener('click', resetToForm);

// ---- Disclaimer modal ----
const _AGREED_KEY = 'nzth_agreed_v1';

function initDisclaimer() {
  if (localStorage.getItem(_AGREED_KEY)) return;
  const modal = document.getElementById('disclaimer-modal');
  const checkbox = document.getElementById('disclaimer-checkbox');
  const agreeBtn = document.getElementById('disclaimer-agree');
  modal.classList.add('visible');
  document.body.classList.add('modal-open');
  checkbox.addEventListener('change', () => {
    agreeBtn.disabled = !checkbox.checked;
  });
  agreeBtn.addEventListener('click', () => {
    localStorage.setItem(_AGREED_KEY, '1');
    modal.classList.remove('visible');
    document.body.classList.remove('modal-open');
  });
}

// ---- Init ----
_loadToken();
_initDebugShortcut();
pollQueue();
setInterval(pollQueue, 15000);
initDisclaimer();

document.getElementById('show-terms').addEventListener('click', (e) => {
  e.preventDefault();
  const modal = document.getElementById('disclaimer-modal');
  modal.classList.add('visible');
  document.body.classList.add('modal-open');
});
