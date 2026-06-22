/**
 * script.js
 * ---------
 * Client-side logic for the Social Media Sentiment Analysis app.
 */

/* ── Helpers ──────────────────────────────────────────────── */

const $ = id => document.getElementById(id);

/** Map a sentiment label to its CSS class and emoji. */
function sentimentMeta(label) {
  const l = (label || '').toLowerCase();
  if (l === 'positive') return { cls: 'positive', emoji: '😊' };
  if (l === 'negative') return { cls: 'negative', emoji: '😞' };
  return { cls: 'neutral', emoji: '😐' };
}

/** Colour for a probability bar given its label. */
function probBarColour(label) {
  const l = (label || '').toLowerCase();
  if (l === 'positive') return 'linear-gradient(90deg,#22c55e,#4ade80)';
  if (l === 'negative') return 'linear-gradient(90deg,#ef4444,#f87171)';
  return 'linear-gradient(90deg,#3b82f6,#60a5fa)';
}

/* ── Character counter ────────────────────────────────────── */

const inputText = $('inputText');
const charCount = $('charCount');

inputText.addEventListener('input', () => {
  charCount.textContent = inputText.value.length;
});

/* ── Example chips ────────────────────────────────────────── */

function loadExample(btn) {
  inputText.value = btn.textContent.trim();
  charCount.textContent = inputText.value.length;
  inputText.focus();
  // Smooth scroll to the input card
  inputText.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* ── Reset ────────────────────────────────────────────────── */

function resetForm() {
  inputText.value = '';
  charCount.textContent = '0';
  $('resultCard').style.display = 'none';
  $('errorCard').style.display = 'none';
  inputText.focus();
}

/* ── Main prediction call ─────────────────────────────────── */

async function analyseText() {
  const text = inputText.value.trim();

  // Basic client-side validation
  if (!text) {
    showError('Please enter some text before analysing.');
    return;
  }

  hideCards();
  showLoading(true);
  $('predictBtn').disabled = true;

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || 'An unknown error occurred.');
      return;
    }

    renderResult(data, text);

  } catch (err) {
    showError('Network error – is the Flask server running?');
    console.error(err);
  } finally {
    showLoading(false);
    $('predictBtn').disabled = false;
  }
}

/* ── Render result ────────────────────────────────────────── */

function renderResult(data, originalText) {
  const { sentiment, confidence, all_probs } = data;
  const meta = sentimentMeta(sentiment);

  // Badge
  const badge = $('sentimentBadge');
  badge.className = `sentiment-badge ${meta.cls}`;
  $('sentimentEmoji').textContent = meta.emoji;
  $('sentimentLabel').textContent = sentiment;

  // Confidence bar
  const bar = $('confidenceBar');
  bar.className = `confidence-bar ${meta.cls}`;
  // Trigger reflow so CSS transition animates from 0
  bar.style.width = '0';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bar.style.width = `${confidence}%`;
    });
  });
  $('confidenceValue').textContent = `${confidence}%`;

  // Probability breakdown
  const probSection = $('probBreakdown');
  const probBars   = $('probBars');
  probBars.innerHTML = '';

  if (all_probs && Object.keys(all_probs).length > 0) {
    probSection.style.display = 'block';
    // Sort descending by probability
    const sorted = Object.entries(all_probs).sort((a, b) => b[1] - a[1]);
    sorted.forEach(([label, prob]) => {
      const pct = (prob * 100).toFixed(1);
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <span class="prob-row-label">${capitalise(label)}</span>
        <div class="prob-row-bar-wrap">
          <div class="prob-row-bar" style="width:0;background:${probBarColour(label)}" data-target="${pct}"></div>
        </div>
        <span class="prob-row-val">${pct}%</span>
      `;
      probBars.appendChild(row);
    });

    // Animate bars after insertion
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        probBars.querySelectorAll('.prob-row-bar').forEach(el => {
          el.style.width = `${el.dataset.target}%`;
        });
      });
    });
  } else {
    probSection.style.display = 'none';
  }

  // Analysed text snippet (truncated for display)
  const snippet = originalText.length > 200
    ? originalText.slice(0, 200) + '…'
    : originalText;
  $('analysedText').textContent = snippet;

  // Show result card
  const card = $('resultCard');
  card.style.display = 'block';
  card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Error display ────────────────────────────────────────── */

function showError(message) {
  $('errorMessage').textContent = message;
  $('errorCard').style.display = 'flex';
  $('errorCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Utility ──────────────────────────────────────────────── */

function hideCards() {
  $('resultCard').style.display = 'none';
  $('errorCard').style.display  = 'none';
}

function showLoading(visible) {
  $('loadingOverlay').style.display = visible ? 'flex' : 'none';
}

function capitalise(str) {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

/* ── Keyboard shortcut: Ctrl/Cmd + Enter ─────────────────── */

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    analyseText();
  }
});
