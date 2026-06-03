'use strict';

// ---- Anonymous session (localStorage UUID, 7-day sliding window) ----
function _getSessionId() {
  const key = 'nz_tenancy_session_id';
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

// ---- Debug mode (#debug to activate) ----
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
      <a class="web-result-link" href="${Astraea.escapeHtml(r.url)}" target="_blank" rel="noopener noreferrer">${Astraea.escapeHtml(r.title)}</a>
      <span class="web-result-domain">${Astraea.escapeHtml(domain)}</span>
      <div class="web-result-body">${Astraea.escapeHtml(r.body)}</div>
    </div>`;
  }).join('');

  const panel = document.createElement('div');
  panel.id = 'web-panel';
  panel.innerHTML = `<h4>Web verify ${badge}</h4>${rows || '<span class="debug-note">no results</span>'}`;
  (container || resultCard).appendChild(panel);
}

// Terms shown in anchor forbidden-term checklist (must match backend _FORBIDDEN_ANCHOR_TERMS).
const _FORBIDDEN_ANCHOR_TERMS_DISPLAY = [
  'Schedule 1A', 'infringement fee', '42A(7)', '19(2)', 'penalty notice',
];

function _buildAnchorCard(s, anchorMethod) {
  const num = (s.document_id || '').replace('NZLEG/RTA/', '');
  const forbidden = s.forbidden_terms || {};
  const checks = _FORBIDDEN_ANCHOR_TERMS_DISPLAY.map(t => {
    const found = forbidden[t];
    return `<span class="${found ? 'ctx-forbidden-fail' : 'ctx-forbidden-ok'}">${Astraea.escapeHtml(t)}: ${found ? 'YES' : 'no'}</span>`;
  }).join(' | ');
  const noText = !s.tokens || s.tokens === 0;
  const card = document.createElement('div');
  card.className = 'ctx-card ctx-card-leg';
  card.innerHTML = `<div class="ctx-card-header">${Astraea.escapeHtml(num)} - ${Astraea.escapeHtml(s.title || '')}</div>
<div class="ctx-card-meta">legislation | ${Astraea.escapeHtml(anchorMethod || '')} | ~${s.tokens ?? '?'} tokens | score: n/a</div>
${noText ? '<div class="ctx-anchor-warn">Warning: anchor section selected but no text extracted - section was not sent to model. Heading pattern may not match or section lacks subsection markers.</div>' : ''}
<div class="ctx-card-forbidden">Forbidden terms: ${checks}</div>
<div class="ctx-card-preview">${Astraea.escapeHtml((s.preview || '').slice(0, 400))}</div>`;
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
  card.innerHTML = `<div class="ctx-card-header">[S${c.source_index}] ${Astraea.escapeHtml(c.document_id || '')}</div>
<div class="ctx-card-meta">case | score: ${Astraea.escapeHtml(scoreStr)} | date: ${Astraea.escapeHtml(c.date || '?')} | ~${c.tokens ?? '?'} tokens${Astraea.escapeHtml(gateMeta)}</div>
<div class="ctx-card-preview">${Astraea.escapeHtml(preview)}</div>
${hasMore ? `<button class="ctx-expand-btn">Show full chunk</button>` : ''}`;
  if (hasMore) {
    const btn = card.querySelector('.ctx-expand-btn');
    btn.dataset.full = fullText;
    btn.dataset.preview = preview;
  }
  return card;
}

// Expand/collapse full chunk text
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
<div class="ctx-budget-detail">Anchor: ~${budget.anchor_tokens} tk | Chunks: ~${budget.chunk_tokens} tk | Sources: ${budget.sources_sent}${Astraea.escapeHtml(truncNote)}</div>`;
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

  const budgetEl = _buildBudgetMeter(ev.budget);
  if (budgetEl) body.appendChild(budgetEl);

  const pl = ev.planner || {};
  const rewriteChanged = ev.rewritten_query && ev.rewritten_query !== ev.original_query;
  let qHtml = '<div class="ctx-query-block">';
  qHtml += `<div class="ctx-query-row"><span class="ctx-label">Original query</span><span class="ctx-query-text">${Astraea.escapeHtml(ev.original_query || '')}</span></div>`;
  if (ev.rewritten_query !== undefined) {
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Rewritten query</span><span class="ctx-query-text${rewriteChanged ? ' ctx-rewrite-changed' : ''}">${Astraea.escapeHtml(ev.rewritten_query || ev.original_query || '')}</span></div>`;
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Rewrite</span><span class="ctx-meta-val">${ev.rewrite_used ? 'yes' : 'no'}</span></div>`;
  }
  if (pl.property_change_triggered) {
    const sections = (pl.forced_sections || []).map(s => s.replace('NZLEG/RTA/', '')).join(', ');
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Prop-change gate</span><span class="ctx-meta-val ctx-gate-yes">triggered | terms: ${Astraea.escapeHtml((pl.trigger_terms || []).join(', '))} | forced: ${Astraea.escapeHtml(sections)}</span></div>`;
    const gate = pl.gate || {};
    const fallbackNote = gate.fallback_used ? ' | FALLBACK: all filtered, using original' : '';
    const rejectedNote = gate.rejected && gate.rejected.length ? ` | rejected: ${Astraea.escapeHtml(gate.rejected.join(', '))}` : '';
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Gate filter</span><span class="ctx-meta-val">before: ${gate.candidates_before ?? '?'} | survived: ${gate.survived ?? '?'}${Astraea.escapeHtml(fallbackNote)}${rejectedNote}</span></div>`;
  }
  const sr = ev.statute_routing || {};
  if (sr.triggered) {
    const routes = (sr.matched_routes || []).join(', ');
    const injected = (sr.forced_sections || []).map(s => s.replace('NZLEG/RTA/', '')).join(', ') || 'none';
    const terms = (sr.trigger_terms || []).join(', ');
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Statute routing</span><span class="ctx-meta-val ctx-gate-yes">routes: ${Astraea.escapeHtml(routes)} | terms: ${Astraea.escapeHtml(terms)} | injected: ${Astraea.escapeHtml(injected)}</span></div>`;
    if (sr.suppressed_sections && sr.suppressed_sections.length) {
      const sup = sr.suppressed_sections.map(s => s.section.replace('NZLEG/RTA/', '')).join(', ');
      qHtml += `<div class="ctx-query-row"><span class="ctx-label">Suppressed</span><span class="ctx-meta-val">${Astraea.escapeHtml(sup)}</span></div>`;
    }
  } else if (ev.statute_routing !== undefined) {
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Statute routing</span><span class="ctx-meta-val">no route matched - leg_store vector only</span></div>`;
  }
  qHtml += '</div>';
  const qBlock = document.createElement('div');
  qBlock.innerHTML = qHtml;
  body.appendChild(qBlock.firstElementChild);

  const anchor = ev.anchor || {};
  if (anchor.sections && anchor.sections.length) {
    const lbl = document.createElement('div');
    lbl.className = 'ctx-section-label';
    lbl.textContent = `RTA anchor - ${anchor.method || 'unknown'} (legislation, not [SN] cited)`;
    body.appendChild(lbl);
    anchor.sections.forEach(s => body.appendChild(_buildAnchorCard(s, anchor.method)));
  }

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
  qHtml += `<div class="ctx-query-row"><span class="ctx-label">Original query</span><span class="ctx-query-text">${Astraea.escapeHtml(ev.original_query || '')}</span></div>`;
  qHtml += `<div class="ctx-query-row"><span class="ctx-label">Rewrite</span><span class="ctx-meta-val">disabled in compare mode - all strategies use the raw query</span></div>`;
  if (pl.property_change_triggered) {
    const sections = (pl.forced_sections || []).map(s => s.replace('NZLEG/RTA/', '')).join(', ');
    qHtml += `<div class="ctx-query-row"><span class="ctx-label">Prop-change gate</span><span class="ctx-meta-val ctx-gate-yes">triggered | terms: ${Astraea.escapeHtml((pl.trigger_terms || []).join(', '))} | forced: ${Astraea.escapeHtml(sections)}</span></div>`;
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

// ---- Token + DOM refs ----
let _apiToken = '';

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

const thumbUp = document.getElementById('thumb-up');
const thumbDown = document.getElementById('thumb-down');
const feedbackComment = document.getElementById('feedback-comment');
const feedbackText = document.getElementById('feedback-text');
const feedbackSubmit = document.getElementById('feedback-submit');
const feedbackThanks = document.getElementById('feedback-thanks');

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
  await Astraea.pollQueue(queueNotice);
}

// ---- Source rendering (tenancy-specific labels + legislation toggle) ----
function renderSources(sources, legislation) {
  Astraea.renderSources(sources, legislation, {
    legislationGroupLabel: 'Relevant legislation',
    decisionGroupLabel: 'Tribunal decisions',
    decisionLabel: 'Tenancy Tribunal',
    showLegToggle: true,
  });
}

function renderConfidence(ev) {
  Astraea.renderConfidence(ev, resultCard);
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
      <summary class="verification-ref"><a href="${Astraea.escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${Astraea.escapeHtml(s.reference)}</a> - click to expand</summary>
      <pre class="verification-excerpt">${Astraea.escapeHtml(s.excerpt)}</pre>
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
    panel.innerHTML = `<div class="leg-cases-label">Decisions citing s${Astraea.escapeHtml(sectionNum)}</div>` +
      cases.map(c => {
        const url = (c.url || '').startsWith('https://') ? c.url : '#';
        const date = c.date || '';
        const n = c.mentions || 1;
        return `<div class="leg-case-row"><a href="${Astraea.escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${Astraea.escapeHtml(date)}</a><span class="leg-case-mentions">${n}x</span></div>`;
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
    Astraea.saveFullFeedback(_artifact, -1, '', false, _apiToken);
  }
}

thumbUp.addEventListener('click', () => submitFeedback(1));
thumbDown.addEventListener('click', () => submitFeedback(-1));
document.getElementById('debug-capture').addEventListener('click', async () => {
  const btn = document.getElementById('debug-capture');
  await Astraea.saveFullFeedback(_artifact, 0, '', true, _apiToken);
  btn.classList.add('saved');
  btn.title = 'Debug context saved';
  setTimeout(() => { btn.classList.remove('saved'); btn.title = 'Save debug context for analysis'; }, 2000);
});

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
  } catch (_) {}
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
  answerBody.innerHTML = Astraea.renderAnswer(fullText);
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

// ---- Compare mode ----
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
  answer.innerHTML = Astraea.renderAnswer(answer.textContent);
  _colAddFeedback(col, strategy);
}

function _colAddFeedback(col, strategy) {
  const fb = document.createElement('div');
  fb.className = 'col-feedback';
  col.appendChild(fb);

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
    Astraea.saveFullFeedback(_colArtifacts[strategy] || { question: currentQuestion, strategy }, -1, '', false, _apiToken);
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
  details.innerHTML = `<summary>Reasoning <span class="think-len">${text.length} chars</span></summary><pre>${Astraea.escapeHtml(text)}</pre>`;
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
      return `<div class="compare-source-row"><span class="source-num source-num--leg">&sect;</span> <a href="${Astraea.escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${Astraea.escapeHtml(s.title || s.case_id)}</a></div>`;
    }).join('');
  }
  if (hasDec) {
    if (hasLeg) html += '<div class="compare-sources-group">Decisions</div>';
    html += sources.map((s, i) => {
      const label = s.date ? `${s.court_name || 'Tribunal'} - ${s.date}` : (s.court_name || 'Tribunal');
      const url = (s.url || '').startsWith('https://') ? s.url : '#';
      return `<div class="compare-source-row"><span class="source-num">S${i+1}</span> <a href="${Astraea.escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${Astraea.escapeHtml(label)}</a></div>`;
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
  answer.innerHTML = `<span class="compare-error">${Astraea.escapeHtml(msg)}</span>`;
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

  const colAnswers = {};

  try {
    await Astraea.streamEvents(res, ev => {
      const s = ev.strategy;
      if (ev.type === 'col_start') {
        _colSetActive(s);
        colAnswers[s] = '';
        _colArtifacts[s] = {
          question, strategy: s,
          irac: document.getElementById('irac-toggle').checked,
          think: document.getElementById('think-toggle').checked,
          debug_mode: true,
          ts_start: new Date().toISOString(), ts_end: null,
          user_agent: navigator.userAgent,
          viewport: { w: window.innerWidth, h: window.innerHeight },
          answer: '', sources: [], legislation: [],
          confidence: null, web_results: null, verification: null,
          debug: null, debug_timing: null, context_debug: null,
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
    });
  } catch (_) {
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
        session_id: _getSessionId(),
        debug_key: _debugKey,
        strategy: strategies[0] || 'vector',
        irac: document.getElementById('irac-toggle').checked,
        feedback_context: true,
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

  let rawAnswer = '';
  let streamedSources = [];
  let streamedLegislation = [];
  let streamingStarted = false;

  try {
    await Astraea.streamEvents(res, event => {
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
      } else if (event.type === 'queue') {
        loadingText.textContent = `Position ${event.position} in queue - estimated wait ~${event.estimated_wait_s}s`;
      } else if (event.type === 'context_debug') {
        _artifact.context_debug = event;
        if (_debugMode) _renderContextDebugPanel(event, resultCard);
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
      }
    });
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
document.getElementById('show-terms').addEventListener('click', (e) => {
  e.preventDefault();
  const modal = document.getElementById('disclaimer-modal');
  modal.classList.add('visible');
  document.body.classList.add('modal-open');
});

// ---- Init ----
Astraea.loadToken().then(t => { _apiToken = t; });
_initDebugShortcut();
pollQueue();
setInterval(pollQueue, 15000);
Astraea.initDisclaimer('nzth_agreed_v1');
