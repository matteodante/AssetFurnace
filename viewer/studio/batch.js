// Batches: run a list of assets from a JSON or CSV spec, follow every row, resume a stopped
// batch, retry the rows that failed. A sheet over the viewer, like Activity. The runner is
// `pipeline.py --run-batch`, started by viewer/batch_api.py; this page only reads its state.
// The pure parts come first, with no DOM, so Node tests import this exact file.

import { closeSheet } from './pages.js';

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const STATUS_LABELS = { queued: 'Waiting', running: 'Running', done: 'Done', failed: 'Failed' };

/** A row's status in words. */
export function statusLabel(status) {
  return STATUS_LABELS[status] || status;
}

/** "7 of 10 done · 1 failed · 2 to go · running now": a batch in one line. */
export function summaryLine({ total = 0, counts = {}, active = null }) {
  const left = (counts.queued || 0) + (counts.running || 0);
  const parts = [`${counts.done || 0} of ${total} done`];
  if (counts.failed) parts.push(`${counts.failed} failed`);
  if (left) parts.push(`${left} to go`, active ? 'running now' : 'stopped');
  return parts.join(' · ');
}

/** Which buttons a batch gets: Stop while it runs; else Resume (rows left) and Retry failed. */
export function batchActions({ counts = {}, active = null }) {
  const left = (counts.queued || 0) + (counts.running || 0);
  return { stop: Boolean(active), resume: !active && left > 0, retry: !active && (counts.failed || 0) > 0 };
}

/** Where a finished row stands in the review queue of the checks. */
export function reviewLabel(review) {
  if (!review) return '';
  if (review.event === 'sent_back') return `Sent back: ${review.note || 'no note'}`;
  return { queued: 'In review', approved: 'Approved' }[review.event] || review.event;
}

export function isSpecFile(name) {
  return /\.(json|csv)$/i.test(name || '');
}

/** The request body for Check or Start. The intent fields only go with a CSV: a JSON spec
 *  states its own, and the server refuses both at once. */
export function submitPayload({ path = '', file = null, useCase = '', distribution = '', commercial = false, checkOnly = false }) {
  const payload = file ? { filename: file.name, content: file.text } : { path: path.trim() };
  if (/\.csv$/i.test(file ? file.name : path.trim())) {
    if (useCase) payload.use_case = useCase;
    if (distribution) payload.distribution = distribution;
    if (commercial) payload.commercial = true;
  }
  if (checkOnly) payload.check_only = true;
  return payload;
}

// ------------------------------------------------------------------------------- the page

const HEADERS = { 'Content-Type': 'application/json', 'X-AssetFurnace-Request': 'batch' };

async function post(url, body = {}) {
  const response = await fetch(url, { method: 'POST', headers: HEADERS, body: JSON.stringify(body) });
  return { ok: response.ok, body: await response.json().catch(() => ({})) };
}

const STYLE = `
.batch-form { display: grid; gap: 8px; padding: 12px; border: 1px solid var(--line); border-radius: 10px; }
.batch-form .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.batch-form input[type=text] { flex: 1 1 320px; min-width: 0; }
.batch-problems { margin: 0; padding-left: 18px; color: var(--ember); }
.batch-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.batch-table th, .batch-table td { text-align: left; padding: 6px 8px; border-top: 1px solid var(--line); vertical-align: top; }
.batch-table .why { white-space: pre-wrap; max-height: 7.5em; overflow: auto; display: block; color: var(--ash); }
.batch-status.failed { color: var(--ember); }
.batch-status.done { color: var(--ok, #7fd18b); }
`;

/** Open the Batches sheet. `open(file)` shows a finished row's GLB, a path under output/. */
export async function openBatches(stage, { open = () => {} } = {}) {
  closeSheet(stage);
  const el = document.createElement('div');
  el.className = 'sheet';
  el.innerHTML = `<style>${STYLE}</style><div class="sheet-card">
    <div class="sheet-top"><h2>Batches</h2><button type="button" class="ghost" data-close>Close</button></div>
    <p class="meta">A batch is a list of assets, one row each, in a JSON or CSV spec (see docs/batches.md). Rows run one after the other;
      a stopped batch carries on from where it stopped. Finished rows land in the Library and in the review queue.
      A batch uses the GPU: do not start other generations while one runs.</p>
    <form class="batch-form" data-form>
      <div class="row"><input type="text" name="path" placeholder="Spec on this computer, e.g. C:\\lots\\props.json" aria-label="Spec path">
        <label class="ghost small">or choose a file <input type="file" name="file" accept=".json,.csv" hidden></label></div>
      <div class="row meta" data-chosen hidden></div>
      <div class="row"><span class="meta">For a CSV:</span>
        <select name="useCase" aria-label="Use case"><option value="">showcase</option><option value="game">game</option></select>
        <select name="distribution" aria-label="Distribution"><option value="">private</option><option value="public">public</option><option value="worldwide">worldwide</option></select>
        <label><input type="checkbox" name="commercial"> commercial</label></div>
      <div class="row"><button type="button" class="ghost" data-check>Check</button><button type="submit" class="primary">Start</button></div>
      <div data-result aria-live="polite"></div>
    </form>
    <h3>Your batches</h3>
    <div data-list aria-live="polite"><p class="meta">Loading…</p></div>
    <div data-detail></div></div>`;
  stage.appendChild(el);

  const form = el.querySelector('[data-form]');
  const result = el.querySelector('[data-result]');
  let file = null;
  let selected = null;
  let timer = null;
  let following = 0;   // refreshes to keep doing after a start, while the runner starts up
  const close = () => { clearTimeout(timer); el.remove(); };
  el.querySelector('[data-close]').addEventListener('click', close);
  el.addEventListener('keydown', (event) => { if (event.key === 'Escape') close(); });

  form.file.addEventListener('change', async () => {
    const chosen = form.file.files[0];
    const label = el.querySelector('[data-chosen]');
    file = chosen && isSpecFile(chosen.name) ? { name: chosen.name, text: await chosen.text() } : null;
    label.hidden = !chosen;
    label.textContent = file ? `Chosen: ${file.name}. Its paths must be absolute, or give its path above instead.`
      : 'Choose a .json or .csv spec.';
  });

  const send = async (checkOnly) => {
    const payload = submitPayload({ path: form.path.value, file, useCase: form.useCase.value,
      distribution: form.distribution.value, commercial: form.commercial.checked, checkOnly });
    result.innerHTML = '<p class="meta">Checking…</p>';
    const { ok, body } = await post('/api/batches', payload);
    if (!ok) {
      const problems = body.problems?.length ? body.problems : [body.error || 'The batch could not start.'];
      result.innerHTML = `<ul class="batch-problems">${problems.map((p) => `<li>${escape(p)}</li>`).join('')}</ul>`;
      return;
    }
    result.innerHTML = checkOnly
      ? `<p class="meta">Ready: ${escape(body.rows)} row${body.rows === 1 ? '' : 's'}, batch "${escape(body.name)}".</p>`
      : `<p class="meta">Started "${escape(body.name)}". Follow it below.</p>`;
    if (!checkOnly) { selected = body.name; following = 5; refresh(); }
  };
  el.querySelector('[data-check]').addEventListener('click', () => send(true));
  form.addEventListener('submit', (event) => { event.preventDefault(); send(false); });

  const list = el.querySelector('[data-list]');
  const detail = el.querySelector('[data-detail]');

  async function refresh() {
    clearTimeout(timer);
    if (!el.isConnected) return;
    let batches = [];
    try {
      batches = (await (await fetch('/api/batches')).json()).batches || [];
    } catch {
      list.innerHTML = '<p class="meta">The batches could not be read. Is the studio server running?</p>';
      return;
    }
    list.innerHTML = batches.length ? `<ul class="act-list">${batches.map((b) => {
      const can = batchActions(b);
      return `<li class="act"><span class="act-mark" aria-hidden="true">${b.active ? '▶' : '·'}</span>
        <span class="act-what"><button type="button" class="ghost small" data-open="${escape(b.name)}"><b>${escape(b.name)}</b></button>
          <span class="meta">${escape(summaryLine(b))}</span></span>
        <span class="row">${can.stop ? `<button class="ghost small" data-stop="${escape(b.name)}">Stop</button>` : ''}
          ${can.resume ? `<button class="ghost small" data-resume="${escape(b.name)}">Resume</button>` : ''}
          ${can.retry ? `<button class="ghost small" data-retry="${escape(b.name)}">Retry failed</button>` : ''}</span></li>`;
    }).join('')}</ul>` : '<p class="meta">No batches yet. Start one above.</p>';
    if (selected) await showDetail(selected);
    if (batches.some((b) => b.active) || following > 0) timer = setTimeout(refresh, 3000);
    following = Math.max(0, following - 1);
  }

  async function showDetail(name) {
    const response = await fetch(`/api/batches/${encodeURIComponent(name)}`);
    if (!response.ok) { detail.innerHTML = ''; return; }
    const info = await response.json();
    detail.innerHTML = `<h3>${escape(info.name)}</h3><table class="batch-table"><thead><tr><th>Row</th><th>Kind</th><th>Status</th><th>Review</th><th></th></tr></thead><tbody>
      ${info.rows.map((row) => `<tr><td>${escape(row.id)}</td><td>${escape(row.kind)}</td>
        <td><span class="batch-status ${escape(row.status)}">${escape(statusLabel(row.status))}</span>
          ${row.status === 'failed' && row.reason ? `<span class="why">${escape(row.reason)}</span>` : ''}</td>
        <td>${escape(reviewLabel(row.review))}</td>
        <td>${row.file ? `<button class="ghost small" data-file="${escape(row.file)}">Open</button>` : ''}</td></tr>`).join('')}
      </tbody></table>`;
  }

  el.addEventListener('click', async (event) => {
    const target = event.target.closest('button');
    if (!target) return;
    const { open: name, stop, resume, retry, file: glb } = target.dataset;
    if (name) { selected = name; await showDetail(name); return; }
    if (glb) { open(glb); return; }
    if (!(stop || resume || retry)) return;
    target.disabled = true;
    const which = stop || resume || retry;
    const { ok, body } = stop ? await post(`/api/batches/${encodeURIComponent(stop)}/cancel`)
      : await post(`/api/batches/${encodeURIComponent(which)}/resume`, { retry_failed: Boolean(retry) });
    if (!ok) result.innerHTML = `<ul class="batch-problems">${(body.problems || [body.error]).map((p) => `<li>${escape(p)}</li>`).join('')}</ul>`;
    selected = which;
    following = stop ? 0 : 5;
    refresh();
  });

  el.querySelector('[data-close]').focus();
  refresh();
}
