'use strict';

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

// ---- Character counter ----
questionEl.addEventListener('input', () => {
  const len = questionEl.value.length;
  charCountEl.textContent = len;
  charCountEl.parentElement.classList.toggle('near-limit', len > 1800);
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
      const items = lines
        .filter(l => /^[-*] /.test(l.trim()))
        .map(l => `<li>${l.trim().replace(/^[-*] /, '')}</li>`)
        .join('');
      return `<ul>${items}</ul>`;
    }
    if (lines.some(l => /^\d+\. /.test(l.trim()))) {
      const items = lines
        .filter(l => /^\d+\. /.test(l.trim()))
        .map(l => {
          const m = l.trim().match(/^(\d+)\. (.*)/);
          return m ? `<li value="${m[1]}">${m[2]}</li>` : `<li>${l.trim()}</li>`;
        })
        .join('');
      return `<ol>${items}</ol>`;
    }
    return `<p>${lines.join('<br>')}</p>`;
  }).join('');

  // Apply inline formatting after structure is built
  return html
    .replace(/\[S(\d+)\]/g, '<span class="citation">[S$1]</span>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderSources(sources) {
  if (!sources || sources.length === 0) {
    sourcesSection.classList.remove('visible');
    return;
  }
  sourcesList.innerHTML = sources.map((s, i) => {
    const court = s.court_name || 'Tenancy Tribunal';
    const date = s.date || '';
    const label = date ? `${court} Decision - ${date}` : `${court} Decision`;
    const rawUrl = s.url || '';
    const url = rawUrl.startsWith('https://') ? rawUrl : '#';
    return `
      <div class="source-card">
        <span class="source-num">S${i + 1}</span>
        <div class="source-info">
          <a class="source-title" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>
        </div>
      </div>`;
  }).join('');
  sourcesSection.classList.add('visible');
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

async function submitFeedback(rating) {
  if (currentRating === rating) {
    // Clicking same button again cancels the selection
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

function finaliseResult(fullText, sources) {
  answerBody.innerHTML = renderAnswer(fullText);
  renderSources(sources);
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
  questionEl.value = '';
  charCountEl.textContent = '0';
  questionEl.focus();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ---- Form submit (SSE streaming) ----
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  if (!question) { questionEl.focus(); return; }
  currentQuestion = question;
  showLoading();

  let res;
  try {
    res = await fetch('/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': _apiToken },
      body: JSON.stringify({ question }),
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
          renderSources(streamedSources);
        } else if (event.type === 'token') {
          if (!streamingStarted) {
            streamingStarted = true;
            showStreamingResult();
            answerBody.textContent = '';
          }
          rawAnswer += event.text;
          answerBody.textContent = rawAnswer;
        } else if (event.type === 'done') {
          finaliseResult(rawAnswer, streamedSources);
        } else if (event.type === 'error') {
          showError(event.message || 'An error occurred.');
          return;
        }
      }
    }
    // If stream ended without a 'done' event, finalise what we have
    if (streamingStarted && rawAnswer) {
      finaliseResult(rawAnswer, streamedSources);
    }
  } catch (_) {
    showError('Lost connection while receiving the answer. Please try again.');
  }
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
pollQueue();
setInterval(pollQueue, 15000);
initDisclaimer();

document.getElementById('show-terms').addEventListener('click', (e) => {
  e.preventDefault();
  const modal = document.getElementById('disclaimer-modal');
  modal.classList.add('visible');
  document.body.classList.add('modal-open');
});
