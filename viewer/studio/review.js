// Review: the queue sheet, and the panel that replaces Steps while an asset is reviewed.
// The main viewer turns the asset 360 degrees; beside the checks, a small viewer turns the
// reference the same way, and the silhouette overlays show where the two outlines differ.

import { createModelViewport, disposeModelViewport } from '../components/model-viewport.js';
import { closeSheet } from './pages.js';
import {
  STATE_LABELS, STATUS_LABELS, statusLabel, queueSections, summaryLine, checkRows, overlays, canDecide, canApprove, approveBlocker,
  checkChoices, checksRequest, thresholdRows,
} from './review-model.js';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const HEADERS = { 'Content-Type': 'application/json', 'X-AssetFurnace-Request': 'review' };

async function getJson(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

async function postJson(url, body) {
  const response = await fetch(url, { method: 'POST', headers: HEADERS, body: JSON.stringify(body) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

export const fetchQueue = () => getJson('/api/review/queue');
export const fetchItem = (key) => getJson(`/api/review/item/${encodeURIComponent(key)}`);

// Two Library assets can share a name (a model and its finished copy): the file tells them apart.
const choiceLabel = (choice) => `${choice.name} · ${choice.version ? `${choice.version} · ` : ''}${choice.file.split('/').pop()}`;

function itemRow(item) {
  const note = item.notes?.length ? `<span class="rv-note">“${esc(item.notes[item.notes.length - 1].note)}”</span>` : '';
  return `<li><button class="rv-item ${esc(item.state)}" data-key="${esc(item.key)}">
    <span class="rv-chip ${esc(item.status)}">${esc(statusLabel(item.status))}</span>
    <span class="rv-what"><b>${esc(item.name || item.key)}</b>
      <span class="meta">${esc(item.key)} · ${esc(summaryLine(item))} · ${esc(String(item.updated_at || '').slice(0, 16).replace('T', ' '))}</span>
      ${note}</span></button></li>`;
}

/** The Review sheet over the viewer: the queue, and a form to check a Library asset. */
export async function openReview(stage, { assets, onPick, onCount }) {
  closeSheet(stage);
  const el = document.createElement('div');
  el.className = 'sheet';
  const choices = checkChoices(assets);
  el.innerHTML = `<div class="sheet-card review">
    <div class="sheet-top"><h2>Review</h2><button type="button" class="ghost" data-close>Close</button></div>
    <p class="meta">Assets wait here after their automatic checks. Open one to turn it round, compare it with its reference, then approve it or send it back with a note. A sent-back asset stays here, note in view, until a new version is checked.</p>
    <details class="adv rv-check"${choices.length ? '' : ' hidden'}><summary>Check an asset from the Library</summary>
      <label class="field">Asset<select data-k="asset">${choices.map((c, i) => `<option value="${i}">${esc(choiceLabel(c))}</option>`).join('')}</select></label>
      <label class="field">Compare with<select data-k="reference"><option value="">No reference</option>
        <option value="@picture">Its own picture</option>
        ${choices.map((c, i) => `<option value="${i}">${esc(choiceLabel(c))}</option>`).join('')}</select></label>
      <label class="field">Kind<select data-k="profile"><option value="prop">Prop</option><option value="creature">Creature</option><option value="equipment_part">Equipment part</option><option value="weapon_or_shield">Weapon or shield</option></select></label>
      <label class="field">Height in metres (optional)<input data-k="height" type="number" min="0" step="0.01" placeholder="1.8"></label>
      <details class="adv rv-thresholds"><summary>Thresholds</summary><p class="meta">The profile's values; change one to check against it this time.</p><div data-thresholds class="rv-grid"></div></details>
      <button class="primary" data-run>Run the checks</button><p class="meta" data-status aria-live="polite"></p>
    </details>
    <div class="rv-queue" aria-live="polite"><p class="meta">Loading…</p></div></div>`;
  stage.appendChild(el);
  el.querySelector('[data-close]').addEventListener('click', () => el.remove());
  el.addEventListener('keydown', (e) => { if (e.key === 'Escape') el.remove(); });
  el.querySelector('[data-close]').focus();

  let currentThresholds = [];
  let asked = 0;   // only the latest request draws the grid: a slower earlier one is dropped
  const loadThresholds = async () => {
    const profile = el.querySelector('[data-k="profile"]').value;
    const grid = el.querySelector('[data-thresholds]');
    const mine = ++asked;
    let rows = [];
    try {
      const response = await fetch(`/api/review/thresholds?profile=${encodeURIComponent(profile)}`);
      rows = (await response.json()).thresholds || [];
    } catch { rows = []; }
    if (mine !== asked) return;
    currentThresholds = rows;
    grid.innerHTML = thresholdRows(currentThresholds).map((r) => `<label class="field">${esc(r.label)}<input type="number" step="any" data-threshold="${esc(r.name)}" value="${esc(r.value)}"></label>`).join('')
      || '<p class="meta">The thresholds could not be read.</p>';
  };
  el.querySelector('[data-k="profile"]')?.addEventListener('change', loadThresholds);
  if (choices.length) loadThresholds();

  const box = el.querySelector('.rv-queue');
  const draw = (payload) => {
    onCount?.(payload);
    const sections = queueSections(payload).filter((s) => s.items.length);
    box.innerHTML = sections.length ? sections.map((s) => `<h3>${esc(s.title)}</h3><ul class="rv-list">${s.items.map(itemRow).join('')}</ul>`).join('')
      : '<p class="meta">Nothing waits for review. Check an asset above, or run a batch with checks.</p>';
  };
  box.addEventListener('click', (e) => {
    const key = e.target.closest('[data-key]')?.dataset.key;
    if (key) { el.remove(); onPick(key); }
  });
  el.querySelector('[data-run]')?.addEventListener('click', async (e) => {
    const value = (k) => el.querySelector(`[data-k="${k}"]`).value;
    const choice = choices[Number(value('asset'))];
    const ref = value('reference');
    const reference = ref === '@picture' ? choice.picture || '' : ref === '' ? '' : choices[Number(ref)].file;
    const status = el.querySelector('[data-status]');
    e.target.disabled = true;
    status.textContent = 'Checking… a few seconds, up to a minute for a raw model.';
    try {
      const edited = Object.fromEntries([...el.querySelectorAll('[data-threshold]')].map((i) => [i.dataset.threshold, i.value]));
      const item = await postJson('/api/review/checks', checksRequest(choice, { reference, profile: value('profile'), height: value('height'),
        thresholds: currentThresholds, edited }));
      el.remove();
      onPick(item.key);
    } catch (error) {
      status.textContent = `The checks did not run: ${error.message}`;
      e.target.disabled = false;
    }
  });
  try { draw(await fetchQueue()); }
  catch (error) { box.innerHTML = `<p class="meta">The queue could not be read: ${esc(error.message)}</p>`; }
}

/** A small viewer that turns the reference model, for as long as its pane is on the page. */
function referenceViewer(pane, url) {
  const view = createModelViewport({ pane, spec: { url, ext: 'glb', label: 'reference' },
    onLoaded: (v) => { v.controls.autoRotate = true; v.controls.autoRotateSpeed = 4; v.camera.position.set(0, 0.15, 2.1); } });
  const fit = () => {
    const { clientWidth: w, clientHeight: h } = pane;
    if (!w || !h) return;
    view.renderer.setSize(w, h);
    view.camera.aspect = w / h;
    view.camera.updateProjectionMatrix();
  };
  fit();
  const frame = () => {
    if (view.disposed) return;
    if (!pane.isConnected) { disposeModelViewport(view); return; }
    fit();
    view.controls.update();
    view.renderer.render(view.scene, view.camera);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function referenceBlock(item) {
  const reference = item.report?.reference;
  if (!reference?.url) return '<p class="meta">No reference given: size and silhouette were not compared.</p>';
  if (reference.kind === 'image') return `<figure class="rv-ref"><img src="${esc(reference.url)}" alt="Reference picture"><figcaption class="meta">Reference picture</figcaption></figure>`;
  return '<figure class="rv-ref"><div class="rv-refpane" data-refpane><span class="stats" hidden></span></div><figcaption class="meta">Reference model, turning</figcaption></figure>';
}

// Only a version waiting for review can be decided: a sent-back one waits for its replacement.
function decisionBlock(item) {
  if (item.state === 'sent_back') return '<p class="meta">Sent back. A new version comes back here once it has been checked.</p>';
  if (item.state !== 'to_review') return `<p class="meta">${esc(STATE_LABELS[item.state] || item.state)}.</p>`;
  return `<label class="field">Note<textarea data-note rows="3" placeholder="What to change (needed to send back)"></textarea></label>
    <div class="row"><button class="primary" data-decide="approved"${canApprove(item) ? '' : ' disabled'}>Approve</button><button class="ghost" data-decide="sent_back" disabled>Send back</button></div>
    ${canApprove(item) ? '' : `<p class="meta">${esc(approveBlocker(item))}</p>`}`;
}

/** The review panel in the Steps column for one asset; `viewer` is the studio viewer. */
export function renderReviewPanel(root, item, { viewer, onDecided }) {
  const shots = overlays(item.report);
  const last = item.notes?.[item.notes.length - 1];
  root.innerHTML = `<li class="rv-panel">
    <div class="rv-head"><span class="rv-chip ${esc(item.status)}">${esc(statusLabel(item.status))}</span>
      <b>${esc(STATE_LABELS[item.state] || item.state)}</b><span class="meta">${esc(item.key)} · version ${esc(String(item.asset_sha256 || '').slice(0, 8))}</span></div>
    ${last ? `<p class="rv-last"><span class="meta">Last note (${esc(last.decision === 'sent_back' ? 'sent back' : 'approved')}):</span> ${esc(last.note)}</p>` : ''}
    <label class="check"><input type="checkbox" data-turn checked><span>Turn 360°</span></label>
    <h3>Reference</h3>${referenceBlock(item)}
    ${shots.length ? `<div class="rv-shots">${shots.map((s) => `<figure><img src="${esc(s.url)}" alt="${esc(s.label)} silhouette"><figcaption class="meta">${esc(s.label)} ${s.score != null ? Number(s.score).toFixed(2) : ''}</figcaption></figure>`).join('')}</div>
      <p class="meta rv-legend"><i class="ref"></i>reference only <i class="mine"></i>this asset only <i class="both"></i>both</p>` : ''}
    <h3>Checks</h3>
    <ul class="rv-checks">${checkRows(item.report).map((r) => `<li class="${esc(r.status)}"><span class="rv-chip ${esc(r.status)}">${esc(STATUS_LABELS[r.status] || 'Skip')}</span><span><b>${esc(r.check)}</b> <span class="meta">${esc(r.message)}</span></span></li>`).join('') || '<li class="meta">No checks report found.</li>'}</ul>
    <h3>Decision</h3>
    ${decisionBlock(item)}
    <p class="meta" data-status aria-live="polite"></p>
    ${item.history?.length ? `<details class="adv"><summary>History (${item.history.length})</summary><ul class="rv-history">${item.history.map((h) => `<li class="meta">${esc(String(h.at || '').slice(0, 16).replace('T', ' '))} · ${esc(h.event.replace('_', ' '))}${h.note ? ` · “${esc(h.note)}”` : ''}</li>`).join('')}</ul></details>` : ''}
  </li>`;
  const pane = root.querySelector('[data-refpane]');
  if (pane) referenceViewer(pane, item.report.reference.url);
  viewer.setTurntable(true);
  root.querySelector('[data-turn]').addEventListener('change', (e) => viewer.setTurntable(e.target.checked));
  const note = root.querySelector('[data-note]');
  const back = root.querySelector('[data-decide="sent_back"]');
  if (!note) return;  // only a version waiting for review can be decided
  note.addEventListener('input', () => { back.disabled = !canDecide('sent_back', note.value); });
  root.querySelectorAll('[data-decide]').forEach((button) => button.addEventListener('click', async () => {
    const decision = button.dataset.decide;
    if (!canDecide(decision, note.value)) return;
    const status = root.querySelector('[data-status]');
    root.querySelectorAll('[data-decide]').forEach((b) => { b.disabled = true; });
    try {
      const updated = await postJson(`/api/review/item/${encodeURIComponent(item.key)}/decision`, { decision, note: note.value, asset_sha256: item.asset_sha256 });
      renderReviewPanel(root, updated, { viewer, onDecided });
      root.querySelector('[data-status]').textContent = decision === 'approved' ? 'Approved.' : 'Sent back. It stays in the queue with your note.';
      onDecided?.(updated);
    } catch (error) {
      status.textContent = `Not saved: ${error.message}`;
      root.querySelector('[data-decide="approved"]').disabled = !canApprove(item);
      back.disabled = !canDecide('sent_back', note.value);
    }
  }));
}
