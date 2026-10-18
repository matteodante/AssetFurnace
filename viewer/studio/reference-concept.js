// Draw from a reference (function 2): a sheet over the viewer that turns a model or a picture
// into the map a concept is drawn from, runs one of the four ways through the image queue,
// and shows the reference beside each result with its silhouette IoU and a keep/discard.
// The server side is viewer/concept_api.py. The pure parts come first, so Node tests import
// this exact file.

import { followJob, plainError, readableLog } from './jobs.js';
import { escapeHTML as esc } from '../components/html.js';

const HEADERS = { 'X-AssetFurnace-Request': 'reference-concept' };
export const PICTURE = /\.(png|jpe?g|webp)$/i;
export const MODEL = /\.glb$/i;

/** The four ways, with what each needs. Measured on 16/10/2026: see the design, section 10. */
export const WAYS = {
  redraw: {
    label: 'Redraw a reference', hint: 'A new design in the same outline and pose. Colours in words: shown the original, the model copies it.',
    shape: 'A model (its depth is used) or a picture (its lines are used)', accepts: ['model', 'picture'],
    fields: [['subject', 'What it is', 'werewolf', true], ['colours', 'Its colours', 'grey fur, a brown mane and pale claws', true],
      ['parts', 'Parts to redesign (optional)', 'fur tufts, claws, face and markings', false]],
  },
  dress: {
    label: 'Dress the base body', hint: 'Clothes and hair on the body, in its exact T-pose.',
    shape: 'The base body model', accepts: ['model', 'picture'],
    fields: [['outfit', 'The outfit', 'a ranger in a green tunic, leather bracers and boots', true]],
  },
  add_piece: {
    label: 'Add a piece to the body', hint: 'One piece drawn on the body; the rest stays as it is.',
    shape: 'The base body model', accepts: ['model', 'picture'],
    fields: [['piece', 'The piece', 'a steel helmet with gold trim', true]],
  },
  other_view: {
    label: 'Another view', hint: 'The same creature from the side or the back. Give its model too, and the outline is followed.',
    shape: 'The front picture of the creature', accepts: ['picture'],
    fields: [],
  },
};

/** What to prepare on the server for a way: one entry per shape picture, in order. */
export function mapsFor(way, source, { guide = 'body', view = 'side', model = null } = {}) {
  if (!source) return [];
  const isModel = MODEL.test(source);
  if (way === 'redraw') return [{ source, kind: isModel ? 'depth' : 'lineart', view: 'front' }];
  if (way === 'dress' && guide === 'pose') return isModel ? [{ source, kind: 'pose', view: 'front' }] : [];
  if (way === 'dress' || way === 'add_piece') return [{ source, kind: isModel ? 'colour' : 'picture', view: 'front' }];
  const front = [{ source, kind: 'picture', view: 'front' }];
  return model ? [...front, { source: model, kind: 'depth', view }] : front;
}

/** Why the sheet cannot start yet, in words; empty when it can. */
export function problemsOf(state) {
  const way = WAYS[state.way];
  const out = [];
  if (!way) return ['Choose a way first.'];
  if (!state.source) out.push(`Choose ${way.shape.toLowerCase()}.`);
  else if (!way.accepts.includes(MODEL.test(state.source) ? 'model' : 'picture')) out.push(`${way.label} starts from ${way.shape.toLowerCase()}.`);
  if (state.way === 'dress' && state.guide === 'pose' && state.source && !MODEL.test(state.source)) out.push('A pose map is drawn from a rigged model, not a picture.');
  for (const [key, label, , required] of way.fields) if (required && !String(state[key] || '').trim()) out.push(`${label} is needed.`);
  if ((state.styleRefs || []).length > 3) out.push('At most 3 style pictures.');
  return out;
}

/** The body of POST /api/concept/jobs for one attempt. */
export function requestFor(state, shape, seed) {
  const body = { mode: state.way, shape, style_refs: state.styleRefs || [], seed };
  if (String(state.style || '').trim()) body.style = state.style.trim();
  if (state.useBible) body.bible = '-';   // the active style bible: its words and pictures, opt-in
  if (state.way === 'redraw') Object.assign(body, { subject: state.subject, colours: state.colours, parts: state.parts || '', structure: MODEL.test(state.source) ? 'depth' : 'lineart' });
  if (state.way === 'dress') Object.assign(body, { outfit: state.outfit, guide: state.guide || 'body' });
  if (state.way === 'add_piece') body.piece = state.piece;
  if (state.way === 'other_view') Object.assign(body, { view: state.view || 'side', depth: shape.length > 1 });
  return body;
}

/** One seed per attempt, consecutive, so a good one can be found again. */
export function seedsFor(attempts, random = Math.random) {
  const count = Math.max(1, Math.min(4, Math.round(Number(attempts)) || 1));
  const first = Math.floor(random() * (2147483647 - count));
  return Array.from({ length: count }, (_, i) => first + i);
}

/**
 * Everything one press of Draw asks for, fixed when it is pressed: the maps, the seeds and
 * the request. The controls stay live while the map is prepared and the tries run, so
 * reading them later would mix two requests (the back's depth with a side prompt).
 */
export function planDraw(state) {
  const asked = { ...state, styleRefs: [...(state.styleRefs || [])] };
  return { maps: mapsFor(asked.way, asked.source, asked), seeds: seedsFor(asked.attempts), request: (shape, seed) => requestFor(asked, shape, seed) };
}

/** The error line, kept in the state so redrawing the sheet does not wipe it. */
export function errorLine(state) {
  return state.error ? `<p class="error" data-error>${esc(state.error)}</p>` : '<p class="error" data-error hidden></p>';
}

/** How the IoU reads under a result. */
export function iouText(record, measuring) {
  if (measuring) return 'measuring the outline…';
  if (record?.iou == null) return 'no outline to compare';
  return `outline IoU ${Number(record.iou).toFixed(2)}`;
}

/** The pictures and models the Library offers as a source, newest first. */
export function sourcesFrom(assets = []) {
  const out = [];
  for (const asset of assets) {
    const model = asset.rigged || asset.finished || asset.model;
    if (model) out.push({ path: model, label: `${asset.name} (model)` });
    if (asset.picture) out.push({ path: asset.picture, label: `${asset.name} (picture)` });
  }
  return out;
}

async function send(url, body, { raw = false } = {}) {
  const response = await fetch(url, { method: 'POST', headers: raw ? HEADERS : { ...HEADERS, 'Content-Type': 'application/json' }, body: raw ? body : JSON.stringify(body) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

const served = (base, path) => base + path.split('/').map(encodeURIComponent).join('/');

/** Show the sheet over the viewer. `onDone()` runs after a picture is made, to refresh the Library. */
export function openReferenceConcept({ stage, assets = [], base = '/output/', styleBible = null, onDone }) {
  const sheet = document.createElement('div');
  sheet.className = 'sheet reference-concept';
  stage.appendChild(sheet);
  const state = { way: 'redraw', source: '', model: '', guide: 'body', view: 'side', attempts: 1, style: '', styleRefs: [], uploads: [], runs: [], busy: false, info: null, error: '', note: '' };
  const close = () => { if (!state.busy) sheet.remove(); };
  fetch('/api/concept').then((r) => r.json()).then((info) => { state.info = info; render(); }).catch(() => {});

  const options = (filter, chosen) => {
    const list = [...state.uploads, ...sourcesFrom(assets)].filter((s) => filter(s.path));
    return `<option value="">Choose…</option>${list.map((s) => `<option value="${esc(s.path)}" ${s.path === chosen ? 'selected' : ''}>${esc(s.label)}</option>`).join('')}`;
  };

  const render = () => {
    const way = WAYS[state.way];
    const ready = state.info?.ready;
    const missing = ready ? ['weights', 'vision', 'binary'].filter((k) => !ready[k]) : [];
    sheet.innerHTML = `<form class="sheet-card" novalidate>
      <div class="sheet-top"><h2>Draw from a <span>reference</span></h2><button type="button" class="ghost" data-close ${state.busy ? 'disabled' : ''}>Close</button></div>
      ${missing.length ? '<p class="warning">The picture model or its vision file is not installed yet. Install Qwen-Image from Setup first.</p>' : ''}
      <div class="q"><span class="qn">1</span><b>Way</b></div>
      <div class="seg ways">${Object.entries(WAYS).map(([k, w]) => `<button type="button" aria-pressed="${k === state.way}" data-way="${k}">${w.label}</button>`).join('')}</div>
      <p class="hint">${esc(way.hint)}</p>
      <div class="q"><span class="qn">2</span><b>Shape</b><span class="hint">${esc(way.shape)}</span></div>
      <div class="rc-source">
        <select data-k="source">${options((p) => way.accepts.includes(MODEL.test(p) ? 'model' : 'picture'), state.source)}</select>
        <button type="button" class="ghost" data-upload>Upload a picture</button>
        <input type="file" data-file hidden accept=".png,.jpg,.jpeg,.webp">
      </div>
      ${state.way === 'dress' ? `<label class="field">Follow<select data-k="guide"><option value="body" ${state.guide === 'body' ? 'selected' : ''}>the body render</option><option value="pose" ${state.guide === 'pose' ? 'selected' : ''}>its pose map (a rigged model)</option></select></label>` : ''}
      ${state.way === 'other_view' ? `<label class="field">View<select data-k="view"><option value="side" ${state.view === 'side' ? 'selected' : ''}>Side</option><option value="back" ${state.view === 'back' ? 'selected' : ''}>Back</option></select></label>
        <label class="field">Its model (optional, recommended)<select data-k="model">${options((p) => MODEL.test(p), state.model)}</select><span class="hint">From the back, without the model 1 try in 2 drew the front again.</span></label>` : ''}
      <div class="q"><span class="qn">3</span><b>Words</b></div>
      ${way.fields.map(([key, label, example]) => `<label class="field">${esc(label)}<input data-k="${key}" value="${esc(state[key] || '')}" placeholder="${esc(example)}"></label>`).join('')}
      <details class="adv"><summary>Style (optional)</summary>
        <label class="field">Style words<input data-k="style" value="${esc(state.style)}" placeholder="stylised fantasy game art, hand-painted look"></label>
        ${styleBible ? `<label class="check"><input type="checkbox" data-k="useBible" ${state.useBible ? 'checked' : ''}><span><b>Style bible:</b> ${esc(styleBible)} <span class="hint">its words and up to three of its pictures</span></span></label>` : ''}
        <p class="hint">Style pictures (up to 3) are painted from, never copied: ${state.styleRefs.length ? state.styleRefs.map(esc).join(', ') : 'none'}.</p>
        <div class="row"><select data-k="styleRef">${options((p) => PICTURE.test(p), '')}</select><button type="button" class="ghost" data-clear-style ${state.styleRefs.length ? '' : 'disabled'}>Clear</button></div>
      </details>
      <label class="field">Tries<select data-k="attempts">${[1, 2, 3, 4].map((n) => `<option value="${n}" ${n === Number(state.attempts) ? 'selected' : ''}>${n}</option>`).join('')}</select><span class="hint">Each try takes about half a minute; pick the best below.</span></label>
      ${state.note ? `<p class="hint" data-note>${esc(state.note)}</p>` : ''}${errorLine(state)}
      <div class="row end"><span class="hint">Qwen-Image 2.1 in edit mode, on this machine</span><button type="submit" class="primary" ${state.busy ? 'disabled' : ''}>${state.busy ? 'Drawing…' : 'Draw it'}</button></div>
      <div class="rc-results">${state.runs.map(runCard).join('')}</div>
    </form>`;
    wire();
  };

  const runCard = (run) => `<figure class="rc-run" data-run="${esc(run.id)}">
      <div class="rc-pair">${run.shape.map((p) => `<img src="${esc(served(base, p))}" alt="shape picture">`).join('')}
        ${run.picture ? `<img src="${esc(served(base, run.picture))}" alt="result">` : `<div class="rc-wait">${esc(run.status)}</div>`}</div>
      <figcaption>seed ${run.seed} · ${esc(run.picture ? iouText(run.record, run.measuring) : run.progress || '')}
        ${run.error ? `<span class="error">${esc(run.error)}</span>` : ''}
        ${run.picture ? `<span class="rc-judge"><button type="button" data-judge="keep" ${run.verdict === 'keep' ? 'aria-pressed="true"' : ''}>Keep</button><button type="button" data-judge="discard" ${run.verdict === 'discard' ? 'aria-pressed="true"' : ''}>Discard</button></span>` : ''}
      </figcaption></figure>`;

  const fail = (message) => { state.error = message; sheet.querySelector('[data-error]').outerHTML = errorLine(state); };

  async function upload(file) {
    if (!file || !PICTURE.test(file.name)) return fail('Choose a PNG, JPG or WebP picture.');
    try {
      const saved = await send(`/api/concept/upload?name=${encodeURIComponent(file.name)}`, file, { raw: true });
      state.uploads.unshift({ path: saved.path, label: `${file.name} (uploaded)` });
      state.source = saved.path;
      render();
    } catch (error) { fail(error.message); }
  }

  async function draw() {
    const problems = problemsOf(state);
    if (problems.length) return fail(problems.join(' '));
    const plan = planDraw(state);
    state.busy = true; state.error = ''; render();
    try {
      const shape = [];
      for (const map of plan.maps) {
        state.note = map.kind === 'picture' ? '' : 'Preparing the map… the first time a model is rendered takes about 20 seconds.';
        render();
        shape.push((await send('/api/concept/prepare', map)).path);
      }
      state.note = '';
      for (const seed of plan.seeds) {
        const run = { id: `${seed}`, seed, shape, status: 'waiting', progress: '', picture: null, record: null };
        state.runs.unshift(run); render();
        const started = await send('/api/concept/jobs', plan.request(shape, seed));
        const final = await followJob(started.status_url, { onUpdate: ({ percent, log }) => {
          run.status = 'drawing'; run.progress = percent == null ? readableLog(log, 1) : `${Math.round(percent)}%`; render();
        } }).done;
        if (final.status !== 'done') { run.error = plainError(final.error || final.log_tail); run.status = final.status; render(); continue; }
        run.picture = final.picture;
        for (let i = 0; i < 20; i += 1) {   // the record, then its IoU, follow the picture by a few seconds
          const found = await fetch(`/api/concept/record?picture=${encodeURIComponent(run.picture)}`).then((r) => r.json()).catch(() => null);
          run.record = found?.record; run.measuring = found?.measuring || !found?.record;
          render();
          if (!run.measuring) break;
          await new Promise((resolve) => setTimeout(resolve, 1500));
        }
        onDone?.();
      }
    } catch (error) { state.error = plainError(error.message); }
    state.busy = false; state.note = ''; render();
  }

  function wire() {
    const q = (sel) => sheet.querySelector(sel);
    q('[data-close]').onclick = close;
    sheet.querySelectorAll('[data-way]').forEach((b) => b.onclick = () => { state.way = b.dataset.way; state.source = ''; state.attempts = state.way === 'other_view' ? 2 : 1; render(); });
    sheet.querySelectorAll('[data-k]').forEach((el) => el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', () => {
      const key = el.dataset.k;
      if (key === 'styleRef') { if (el.value && !state.styleRefs.includes(el.value)) state.styleRefs = [...state.styleRefs, el.value].slice(0, 3); render(); return; }
      state[key] = el.type === 'checkbox' ? el.checked : el.value;
      if (el.tagName === 'SELECT') render();
    }));
    q('[data-clear-style]').onclick = () => { state.styleRefs = []; render(); };
    q('[data-upload]').onclick = () => q('[data-file]').click();
    q('[data-file]').onchange = (e) => upload(e.target.files[0]);
    sheet.querySelectorAll('[data-judge]').forEach((b) => b.onclick = async () => {
      const run = state.runs.find((r) => r.id === b.closest('[data-run]').dataset.run);
      try { await send('/api/concept/judge', { picture: run.picture, verdict: b.dataset.judge }); run.verdict = b.dataset.judge; render(); } catch (error) { fail(error.message); }
    });
    q('form').onsubmit = (event) => { event.preventDefault(); if (!state.busy) draw(); };
  }

  sheet.addEventListener('dragover', (e) => e.preventDefault());
  sheet.addEventListener('drop', (e) => { e.preventDefault(); upload(e.dataTransfer.files[0]); });
  sheet.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  render();
}
