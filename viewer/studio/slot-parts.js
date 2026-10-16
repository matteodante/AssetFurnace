// Characters in pieces (feature 4): choose how to make a character, then the pieces it wears.
// Three modes: one piece (the usual + Create), dressed and split into pieces (a concept drawn
// on a base body, cut into slots), bare body plus equipment (sets drawn on a registered body,
// swapped piece by piece). A sheet over the viewer, like Batches; the work is
// `scripts/slot_parts.py run`, started by viewer/slot_parts_api.py. The pure parts come
// first, with no DOM, so Node tests import this exact file.

import { closeSheet } from './pages.js';

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export const MODES = [
  { id: 1, title: 'One piece', text: 'The whole character as one model, as + Create makes it. Nothing to split.' },
  { id: 2, title: 'Dressed, split into pieces', text: 'A concept of the dressed character, drawn on a base body, cut into the pieces you choose. They fit each other and the body.' },
  { id: 3, title: 'Bare body + equipment', text: 'A base body in a bodysuit, then sets of pieces drawn on it to wear and swap one by one (as in 4Story).' },
];

const SLOT_WORDS = { helmet: 'Helmet', torso: 'Torso', shoulders: 'Shoulders', gloves: 'Gloves', legs: 'Legs',
  boots: 'Boots', weapon: 'Weapon', shield: 'Shield' };

/** A slot's name in words. */
export function slotLabel(name) {
  return SLOT_WORDS[name] || name;
}

/** Library characters that can be a base body: rigged humanoids made from a picture with Pixal3D. */
export function bodyCandidates(assets = []) {
  return assets.filter((a) => a.kind === 'character' && a.rigged && a.picture && a.model
    && /\.glb$/i.test(a.model) && (a.family === undefined || a.family === null || a.family === 'humanoid'));
}

/** "6 pieces · body covered 73% · 2 to look at": a set in one line. */
export function setLine(set = {}) {
  const parts = [`${(set.pieces || []).length} piece${(set.pieces || []).length === 1 ? '' : 's'}`];
  if (typeof set.covered === 'number') parts.push(`body covered ${Math.round(set.covered * 100)}%`);
  const bad = Object.values(set.statuses || {}).filter((s) => s && s !== 'pass').length;
  if (bad) parts.push(`${bad} to look at`);
  return parts.join(' · ');
}

/** The run request from the form's state; throws with what is missing, in words. */
export function runPayload(state = {}) {
  const mode = Number(state.mode);
  if (mode === 1) throw new Error('One piece is the usual + Create: use that button.');
  if (mode !== 2 && mode !== 3) throw new Error('Choose how to make the character.');
  if (!state.body) throw new Error('Choose a base body (register one below if there is none).');
  const name = String(state.set || '').trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(name)) throw new Error("Give the set a name: letters, digits, '.', '_' or '-'.");
  const slots = {};
  for (const [slot, options] of Object.entries(state.slots || {})) {
    if (!options || !options.on || slot === 'weapon' || slot === 'shield') continue;
    slots[slot] = { route: options.route === 'B0' ? 'B0' : 'C', split: slot === 'shoulders' && Boolean(options.split),
      prompt: String(options.prompt || '').trim() };
  }
  const payload = { mode, body: state.body, set: name, slots, seeds: String(state.seeds || '3') };
  for (const held of ['weapon', 'shield']) {
    const options = (state.slots || {})[held];
    if (options && options.on) payload[held] = String(options.prompt || '').trim() || (held === 'weapon' ? 'a one-handed sword' : 'a round shield');
  }
  if (!Object.keys(slots).length && !payload.weapon && !payload.shield) throw new Error('Choose at least one piece.');
  if (mode === 2) {
    if (!state.concept) throw new Error('Give the concept of the dressed character (a picture under output/, drawn on the body).');
    payload.concept = state.concept;
  } else {
    const outfit = String(state.outfit || '').trim();
    if (!outfit) throw new Error('Describe the outfit to draw on the body.');
    payload.outfit = outfit;
  }
  if (String(state.style || '').trim()) payload.style = String(state.style).trim();
  return payload;
}

/** The body registration request for a Library character. */
export function bodyPayload(asset, name, labels = {}) {
  if (!asset) throw new Error('Choose a character from the Library.');
  const clean = String(name || '').trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(clean)) throw new Error("Name the body: letters, digits, '.', '_' or '-'.");
  return { name: clean, picture: asset.picture, model: asset.model, rigged: asset.rigged,
    clips: (asset.clips || []).map((c) => ({ name: c.name, file: c.file })), labels };
}

// ------------------------------------------------------------------------------- the page

const HEADERS = { 'Content-Type': 'application/json', 'X-AssetFurnace-Request': 'slot-parts' };

async function post(url, body = {}) {
  const response = await fetch(url, { method: 'POST', headers: HEADERS, body: JSON.stringify(body) });
  return { ok: response.ok, body: await response.json().catch(() => ({})) };
}

const STYLE = `
.sp-modes { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; }
.sp-modes label { display: block; padding: 10px; border: 1px solid var(--line); border-radius: 10px; cursor: pointer; }
.sp-modes input { margin-right: 6px; }
.sp-form { display: grid; gap: 8px; padding: 12px; border: 1px solid var(--line); border-radius: 10px; margin-top: 10px; }
.sp-form .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.sp-form input[type=text], .sp-form textarea { flex: 1 1 260px; min-width: 0; }
.sp-slots { width: 100%; border-collapse: collapse; font-size: 13px; }
.sp-slots td { padding: 4px 6px; border-top: 1px solid var(--line); }
.sp-slots input[type=text] { width: 100%; }
.sp-problems { color: var(--ember); margin: 0; }
.sp-log { white-space: pre-wrap; font-size: 12px; color: var(--ash); max-height: 9em; overflow: auto; }
`;

/** Open the Characters in pieces sheet. `assets()` is the Library; `open(file, title)` shows a GLB. */
export async function openSlotParts(stage, { assets = () => [], open = () => {}, create = () => {} } = {}) {
  closeSheet(stage);
  const el = document.createElement('div');
  el.className = 'sheet';
  el.innerHTML = `<style>${STYLE}</style><div class="sheet-card">
    <div class="sheet-top"><h2>Characters in pieces</h2><button type="button" class="ghost" data-close>Close</button></div>
    <p class="meta">Each piece gets its own clean picture and its own 3D model, then is fitted to the base body, skinned to its skeleton
      and checked in a walk and a sword swing. A set takes about half an hour of GPU; every piece then waits in Review.</p>
    <div class="sp-modes" role="radiogroup" aria-label="How to make the character">
      ${MODES.map((m) => `<label><input type="radio" name="mode" value="${m.id}" ${m.id === 3 ? 'checked' : ''}><b>${escape(m.title)}</b>
        <span class="meta">${escape(m.text)}</span></label>`).join('')}
    </div>
    <form class="sp-form" data-form><div data-fields></div><div data-result aria-live="polite"></div></form>
    <h3>Base bodies</h3><div data-bodies><p class="meta">Loading…</p></div>
    <details><summary>Register a base body from the Library</summary><div class="sp-form" data-register></div></details>
    <h3>Sets</h3><div data-sets></div>
    <div data-job aria-live="polite"></div></div>`;
  stage.appendChild(el);
  let info = { bodies: [], sets: [], slots: [], job: { running: false } };
  let timer = null;
  const close = () => { clearTimeout(timer); el.remove(); };
  el.querySelector('[data-close]').addEventListener('click', close);
  el.addEventListener('keydown', (event) => { if (event.key === 'Escape') close(); });
  const mode = () => Number(el.querySelector('input[name=mode]:checked')?.value || 3);
  const result = el.querySelector('[data-result]');

  function drawForm() {
    const fields = el.querySelector('[data-fields]');
    if (mode() === 1) {
      fields.innerHTML = `<p>${escape(MODES[0].text)}</p><div class="row"><button type="button" class="primary" data-create>Open + Create</button></div>`;
      fields.querySelector('[data-create]').addEventListener('click', () => { close(); create(); });
      return;
    }
    const bodies = info.bodies.map((b) => `<option value="${escape(b.name)}">${escape(b.name)}${Object.values(b.labels || {}).length ? ` (${escape(Object.values(b.labels).join(', '))})` : ''}</option>`).join('');
    fields.innerHTML = `
      <div class="row"><label>Base body <select name="body">${bodies || '<option value="">none yet</option>'}</select></label>
        <label>Set name <input type="text" name="set" placeholder="red-knight"></label></div>
      ${mode() === 2
        ? '<div class="row"><input type="text" name="concept" placeholder="Concept drawn on that body, a picture under output/ (e.g. images/knight.png)"></div>'
        : '<div class="row"><textarea name="outfit" rows="2" placeholder="The outfit in words, e.g. crimson and gold plate armour: a closed helmet, rounded pauldrons, gauntlets, armoured boots"></textarea></div>'}
      <table class="sp-slots"><tbody>${(info.slots || []).map((s) => `<tr>
        <td><label><input type="checkbox" data-on="${escape(s.name)}"> ${escape(slotLabel(s.name))}</label></td>
        <td>${s.drawn ? `<select data-route="${escape(s.name)}" aria-label="How to make it"><option value="C">own picture</option><option value="B0">cut from the whole</option></select>` : '<span class="meta">made on its own</span>'}</td>
        <td>${s.name === 'shoulders' ? `<label><input type="checkbox" data-split="shoulders"> two pieces</label>` : ''}</td>
        <td><input type="text" data-prompt="${escape(s.name)}" placeholder="${s.held ? 'what it is, e.g. a longsword' : 'optional words for this piece'}"></td></tr>`).join('')}</tbody></table>
      <div class="row"><label>Seeds <input type="text" name="seeds" value="3" size="8"></label>
        <label>Style <input type="text" name="style" placeholder="optional, e.g. hand-painted"></label>
        <button type="submit" class="primary">Make the set</button></div>`;
  }

  function state() {
    const form = el.querySelector('[data-form]');
    const value = (name) => form.querySelector(`[name=${name}]`)?.value || '';
    const slots = {};
    for (const s of info.slots || []) {
      slots[s.name] = { on: form.querySelector(`[data-on="${s.name}"]`)?.checked,
        route: form.querySelector(`[data-route="${s.name}"]`)?.value, split: form.querySelector(`[data-split="${s.name}"]`)?.checked,
        prompt: form.querySelector(`[data-prompt="${s.name}"]`)?.value };
    }
    return { mode: mode(), body: value('body'), set: value('set'), concept: value('concept'), outfit: value('outfit'),
      seeds: value('seeds'), style: value('style'), slots };
  }

  function drawLists() {
    el.querySelector('[data-bodies]').innerHTML = info.bodies.length
      ? `<ul class="act-list">${info.bodies.map((b) => `<li class="act"><span class="act-what"><b>${escape(b.name)}</b>
          <span class="meta">${escape(Object.entries(b.labels || {}).map(([k, v]) => `${k} ${v}`).join(' · '))}${b.clips?.length ? ` · clips: ${escape(b.clips.join(', '))}` : ' · no clips: checks measure the rest pose only'}</span></span></li>`).join('')}</ul>`
      : '<p class="meta">No base body yet. Make a humanoid in a bodysuit with + Create (picture, 3D, rig), then register it below.</p>';
    el.querySelector('[data-sets]').innerHTML = info.sets.length
      ? `<ul class="act-list">${info.sets.map((s) => `<li class="act"><span class="act-what"><b>${escape(s.set)}</b> on ${escape(s.body)}
          <span class="meta">${escape(setLine(s))}</span></span>
          ${s.preview ? `<button type="button" class="ghost small" data-preview="${escape(s.preview)}" data-title="${escape(`${s.set} on ${s.body}`)}">Show</button>` : ''}</li>`).join('')}</ul>`
      : '<p class="meta">No sets yet.</p>';
    const candidates = bodyCandidates(assets());
    el.querySelector('[data-register]').innerHTML = candidates.length
      ? `<div class="row"><select name="asset">${candidates.map((a) => `<option value="${escape(a.id)}">${escape(a.name)}</option>`).join('')}</select>
          <input type="text" name="bodyName" placeholder="body name, e.g. human-male"></div>
        <div class="row"><input type="text" name="bodyLabels" placeholder="labels, e.g. race=human, sex=male">
          <button type="button" class="ghost" data-register-go>Register</button></div><div data-register-result></div>`
      : '<p class="meta">No rigged humanoid in the Library yet.</p>';
    el.querySelector('[data-register-go]')?.addEventListener('click', async () => {
      const box = el.querySelector('[data-register]');
      const out = box.querySelector('[data-register-result]');
      try {
        const asset = candidates.find((a) => a.id === box.querySelector('[name=asset]').value);
        const labels = Object.fromEntries(box.querySelector('[name=bodyLabels]').value.split(',')
          .map((pair) => pair.split('=').map((x) => x.trim())).filter(([k, v]) => k && v));
        const { ok, body } = await post('/api/slot-parts/bodies', bodyPayload(asset, box.querySelector('[name=bodyName]').value, labels));
        out.innerHTML = ok ? `<p class="meta">Registered ${escape(body.name)}.</p>` : `<p class="sp-problems">${escape(body.error)}</p>`;
        if (ok) refresh();
      } catch (error) {
        out.innerHTML = `<p class="sp-problems">${escape(error.message)}</p>`;
      }
    });
  }

  function drawJob() {
    const job = info.job || {};
    const box = el.querySelector('[data-job]');
    if (!job.set) { box.innerHTML = ''; return; }
    const state = job.running ? 'Making' : job.returncode === 0 ? 'Made' : 'Stopped or failed';
    box.innerHTML = `<h3>${escape(state)}: ${escape(job.set)} on ${escape(job.body)}</h3>
      ${job.running ? '<button type="button" class="ghost small" data-cancel>Stop</button>' : ''}
      <div class="sp-log">${escape((job.log || []).join('\n'))}</div>`;
    box.querySelector('[data-cancel]')?.addEventListener('click', async () => { await post('/api/slot-parts/run/cancel'); refresh(); });
  }

  async function refresh() {
    clearTimeout(timer);
    if (!el.isConnected) return;
    try {
      info = await (await fetch('/api/slot-parts')).json();
    } catch {
      el.querySelector('[data-bodies]').innerHTML = '<p class="meta">This page could not reach the studio server.</p>';
      return;
    }
    if (!el.querySelector('[data-fields]').children.length) drawForm();
    drawLists();
    drawJob();
    if (info.job?.running) timer = setTimeout(refresh, 4000);
  }

  el.querySelectorAll('input[name=mode]').forEach((input) => input.addEventListener('change', () => { result.innerHTML = ''; drawForm(); }));
  el.querySelector('[data-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const { ok, body } = await post('/api/slot-parts/run', runPayload(state()));
      result.innerHTML = ok ? '<p class="meta">Started. Follow it below; each piece lands in Review.</p>' : `<p class="sp-problems">${escape(body.error)}</p>`;
      if (ok) refresh();
    } catch (error) {
      result.innerHTML = `<p class="sp-problems">${escape(error.message)}</p>`;
    }
  });
  el.addEventListener('click', (event) => {
    const button = event.target.closest('[data-preview]');
    if (button) { open(button.dataset.preview, button.dataset.title); close(); }
  });
  el.querySelector('[data-close]').focus();
  await refresh();
  drawForm();
}
