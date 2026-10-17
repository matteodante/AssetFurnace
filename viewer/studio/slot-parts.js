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
  { id: 3, title: 'Bare body + equipment', text: 'A base body in the clothes you choose (a bodysuit, simple clothes or underwear), then sets of pieces drawn on it to wear and swap one by one (as in 4Story).' },
];

const SLOT_WORDS = { helmet: 'Helmet', torso: 'Torso', shoulders: 'Shoulders', gloves: 'Gloves', legs: 'Legs',
  boots: 'Boots', weapon: 'Weapon', shield: 'Shield', concept: 'Outfit' };

/** A slot's name in words. */
// The base-body choices; redrawn whenever the list of bodies changes, so a body just
// registered can be picked at once.
export function bodyOptions(bodies) {
  return (bodies || []).map((b) => `<option value="${escape(b.name)}">${escape(b.name)}${Object.values(b.labels || {}).length
    ? ` (${escape(Object.values(b.labels).join(', '))})` : ''}</option>`).join('') || '<option value="">none yet</option>';
}

// One row per piece (or the outfit, `concept`): its candidate pictures, scores underneath
// when there are any, the picked one ticked; a piece no picture convinced the judge of says so.
const scoreText = (score) => (typeof score === 'number' ? ` · ${Math.round(100 * score)}%` : '');

export function pickRows(picks, base = '/output/') {
  return Object.entries(picks || {}).map(([slot, entry]) => `<div class="sp-pick">
    <b>${escape(slotLabel(slot))}</b>${entry.doubt ? ' <span class="sp-problems">none of these looked like the piece: try other seeds or words</span>' : ''}
    <div class="sp-pick-row">${entry.candidates.map((c) => `<label class="sp-cand${c.seed === entry.auto ? ' sp-auto' : ''}">
      <img src="${escape(base + c.picture)}" alt="${escape(slot)} seed ${c.seed}" loading="lazy">
      <span><input type="radio" name="pick-${escape(slot)}" data-pick="${escape(slot)}" value="${c.seed}"${c.seed === entry.picked ? ' checked' : ''}>
      seed ${c.seed}${scoreText(c.score)}${c.seed === entry.auto ? (typeof c.score === 'number' ? ' · best' : ' · first') : ''}</span></label>`).join('')}</div></div>`).join('');
}

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

/** A made set's actions: fit it again, carry it to another body, export it for Unreal. */
export function setActions(set = {}, bodies = []) {
  const others = (bodies || []).filter((b) => b.name !== set.body);
  const data = `data-set="${escape(set.set)}" data-body="${escape(set.body)}"`;
  return `<span class="sp-set-actions">
    <button type="button" class="ghost small" data-set-action="fit" ${data} title="Fit the pieces on the body again, after a piece was changed">Fit again</button>
    ${others.length ? `<select data-refit-to aria-label="Body to carry the set to">${others.map((b) => `<option value="${escape(b.name)}">${escape(b.name)}</option>`).join('')}</select>
    <button type="button" class="ghost small" data-set-action="refit" ${data} title="Carry the fitted pieces to the chosen body">Carry to body</button>` : ''}
    <label class="meta">height cm <input type="number" data-size-cm min="1" max="100000" step="1" value="180" style="width:5em"></label>
    <button type="button" class="ghost small" data-set-action="unreal" ${data} title="The masked body and every piece on its Skeleton, for Unreal 5.8">Export for Unreal</button></span>`;
}

/** A material menu's options: "Auto" (the rule decides) first unless `auto` is null, then the words. */
export function materialOptions(materials = [], value = '', auto = 'Auto') {
  const words = (materials || []).map((m) => `<option value="${escape(m)}"${m === value ? ' selected' : ''}>${escape(m)}</option>`);
  return (auto === null ? '' : `<option value=""${value ? '' : ' selected'}>${escape(auto)}</option>`) + words.join('');
}

/** A made set's pieces, each with a menu on its material and why it has it. */
export function materialRows(pieces = {}, materials = []) {
  return Object.entries(pieces || {}).map(([slot, entry]) => `<tr><td>${escape(slotLabel(slot))}</td>
    <td><select data-set-material="${escape(slot)}" data-was="${escape(entry.material)}" aria-label="${escape(slotLabel(slot))} material">${materialOptions(materials, entry.material, null)}</select></td>
    <td class="meta">${escape(entry.reason || '')}</td></tr>`).join('');
}

/** Only the pieces whose menu a person changed: {slot: material}. */
export function changedMaterials(menus = []) {
  return Object.fromEntries(menus.filter((m) => m.value !== m.was).map((m) => [m.slot, m.value]));
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
    if (options.material) slots[slot].material = String(options.material);
  }
  for (const held of ['weapon', 'shield']) {
    const options = (state.slots || {})[held];
    if (options && options.on && options.material) slots[held] = { material: String(options.material) };
  }
  const payload = { mode, body: state.body, set: name, slots, seeds: String(state.seeds || '3'), choose: Boolean(state.choose) };
  for (const held of ['weapon', 'shield']) {
    const options = (state.slots || {})[held];
    if (options && options.on) payload[held] = String(options.prompt || '').trim() || (held === 'weapon' ? 'a one-handed sword' : 'a round shield');
  }
  if (!Object.keys(slots).some((s) => s !== 'weapon' && s !== 'shield') && !payload.weapon && !payload.shield) {
    throw new Error('Choose at least one piece.');
  }
  if (mode === 2) {
    if (!state.concept) throw new Error('Give the concept of the dressed character (a picture under output/, drawn on the body).');
    payload.concept = state.concept;
  } else {
    const outfit = String(state.outfit || '').trim();
    if (!outfit) throw new Error('Describe the outfit to draw on the body.');
    payload.outfit = outfit;
  }
  if (String(state.style || '').trim()) payload.style = String(state.style).trim();
  if (state.useBible) payload.bible = '-';
  if (mode === 3 && String(state.conceptSeeds || '').trim()) {
    payload.concept_seeds = String(state.conceptSeeds).trim();
    payload.choose_concept = Boolean(state.chooseConcept);
  }
  return payload;
}

// How a base body is dressed under its pieces: the person chooses when registering it, nothing is assumed.
export const DRESSINGS = [['bodysuit', 'A bodysuit (leotard)'], ['clothes', 'Simple clothes: shirt, breeches, cloth shoes'],
  ['underwear', 'Minimal underwear'], ['other', 'Other (my own words)']];

/** The body registration request for a Library character. */
export function bodyPayload(asset, name, labels = {}, dressing = '', dressingWords = '') {
  if (!asset) throw new Error('Choose a character from the Library.');
  const clean = String(name || '').trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(clean)) throw new Error("Name the body: letters, digits, '.', '_' or '-'.");
  const words = String(dressingWords || '').trim();
  if (!DRESSINGS.some(([v]) => v === dressing) || (dressing === 'other' && !words)) {
    throw new Error('Say how the body is dressed under its pieces (and describe it, for other).');
  }
  return { name: clean, picture: asset.picture, model: asset.model, rigged: asset.rigged,
    clips: (asset.clips || []).map((c) => ({ name: c.name, file: c.file })), labels, dressing, dressing_words: words };
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
.sp-pick { margin: 8px 0; }
.sp-pick-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.sp-cand { display: grid; gap: 4px; width: 132px; font-size: 12px; cursor: pointer; }
.sp-cand img { width: 132px; height: 132px; object-fit: contain; border: 1px solid var(--line); border-radius: 8px; background: #d9d9d9; }
.sp-cand.sp-auto img { border-color: var(--flame); }
.sp-cand:has(input:checked) img { outline: 2px solid var(--ember); }
.sp-log { white-space: pre-wrap; font-size: 12px; color: var(--ash); max-height: 9em; overflow: auto; }
[data-sets] .act-what { grid-column: 1 / 3; }
[data-bodies] .act-what { grid-column: 1 / -1; }
.sp-materials { margin: 4px 0 8px; grid-column: 1 / -1; }
.sp-set-actions { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; grid-column: 1 / -1; }
.sp-set-actions select { width: auto; }
.sp-materials table { border-collapse: collapse; font-size: 13px; }
.sp-materials td { padding: 3px 6px; }
`;

/** Open the Characters in pieces sheet. `assets()` is the Library; `open(file, title)` shows a GLB. */
export async function openSlotParts(stage, { assets = () => [], open = () => {}, create = () => {}, styleBible = null } = {}) {
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
    fields.innerHTML = `
      <div class="row"><label>Base body <select name="body" data-names="${escape(info.bodies.map((b) => b.name).join('|'))}">${bodyOptions(info.bodies)}</select></label>
        <label>Set name <input type="text" name="set" placeholder="red-knight"></label></div>
      ${mode() === 2
        ? '<div class="row"><input type="text" name="concept" placeholder="Concept drawn on that body, a picture under output/ (e.g. images/knight.png)"></div>'
        : `<div class="row"><textarea name="outfit" rows="2" placeholder="The outfit in words, e.g. crimson and gold plate armour: a closed helmet, rounded pauldrons, gauntlets, armoured boots"></textarea></div>
          <div class="row"><label title="The outfit drawn once per seed on the body; empty: once, with the first picture seed">Outfit pictures (seeds) <input type="text" name="conceptSeeds" placeholder="first seed" size="8"></label>
            <label><input type="checkbox" name="chooseConcept" checked> let me pick the outfit before the pieces</label></div>`}
      <table class="sp-slots"><tbody>${(info.slots || []).map((s) => `<tr>
        <td><label><input type="checkbox" data-on="${escape(s.name)}"> ${escape(slotLabel(s.name))}</label></td>
        <td>${s.drawn ? `<select data-route="${escape(s.name)}" aria-label="How to make it"><option value="C">own picture</option><option value="B0">cut from the whole</option></select>` : '<span class="meta">made on its own</span>'}</td>
        <td>${s.name === 'shoulders' ? `<label><input type="checkbox" data-split="shoulders"> two pieces</label>` : ''}</td>
        <td><input type="text" data-prompt="${escape(s.name)}" placeholder="${s.held ? 'what it is, e.g. a longsword' : 'optional words for this piece'}"></td>
        <td><select data-material="${escape(s.name)}" aria-label="${escape(slotLabel(s.name))} material" title="One material for the whole piece: Auto lets the rule pick it from the words">${materialOptions(info.materials)}</select></td></tr>`).join('')}</tbody></table>
      <div class="row"><label title="One picture of each piece per seed; the best-scored goes to 3D">Pictures per piece (seeds) <input type="text" name="seeds" value="1,2,3" size="8"></label>
        <label><input type="checkbox" name="choose" checked> let me check the pictures before the 3D</label>
        <label>Style <input type="text" name="style" placeholder="optional, e.g. hand-painted"></label>
        ${styleBible ? `<label title="Its words for every piece"><input type="checkbox" name="useBible"> style bible: ${escape(styleBible)}</label>` : ''}
        <button type="submit" class="primary">Make the set</button></div>`;
  }

  function updateBodies() {
    const select = el.querySelector('[data-fields] select[name=body]');
    const names = info.bodies.map((b) => b.name).join('|');
    if (!select || select.dataset.names === names) return;
    const chosen = select.value;
    select.innerHTML = bodyOptions(info.bodies);
    select.dataset.names = names;
    if (info.bodies.some((b) => b.name === chosen)) select.value = chosen;
  }

  function state() {
    const form = el.querySelector('[data-form]');
    const value = (name) => form.querySelector(`[name=${name}]`)?.value || '';
    const slots = {};
    for (const s of info.slots || []) {
      slots[s.name] = { on: form.querySelector(`[data-on="${s.name}"]`)?.checked,
        route: form.querySelector(`[data-route="${s.name}"]`)?.value, split: form.querySelector(`[data-split="${s.name}"]`)?.checked,
        prompt: form.querySelector(`[data-prompt="${s.name}"]`)?.value, material: form.querySelector(`[data-material="${s.name}"]`)?.value };
    }
    return { mode: mode(), body: value('body'), set: value('set'), concept: value('concept'), outfit: value('outfit'),
      seeds: value('seeds'), style: value('style'), choose: form.querySelector('[name=choose]')?.checked, slots,
      useBible: form.querySelector('[name=useBible]')?.checked, conceptSeeds: value('conceptSeeds'),
      chooseConcept: form.querySelector('[name=chooseConcept]')?.checked };
  }

  function drawLists() {
    el.querySelector('[data-bodies]').innerHTML = info.bodies.length
      ? `<ul class="act-list">${info.bodies.map((b) => `<li class="act"><span class="act-what"><b>${escape(b.name)}</b>
          <span class="meta">${escape(Object.entries(b.labels || {}).map(([k, v]) => `${k} ${v}`).join(' · '))}${b.clips?.length ? ` · clips: ${escape(b.clips.join(', '))}` : ' · no clips: checks measure the rest pose only'}</span></span></li>`).join('')}</ul>`
      : '<p class="meta">No base body yet. Make a humanoid in what it should wear under its armour (a bodysuit, simple clothes or underwear) with + Create (picture, 3D, rig), then register it below.</p>';
    el.querySelector('[data-sets]').innerHTML = info.sets.length
      ? `<ul class="act-list">${info.sets.map((s) => `<li class="act"><span class="act-what"><b>${escape(s.set)}</b> on ${escape(s.body)}
          <span class="meta">${escape(setLine(s))}</span></span>
          ${s.preview ? `<button type="button" class="ghost small" data-preview="${escape(s.preview)}" data-title="${escape(`${s.set} on ${s.body}`)}">Show</button>` : ''}
          ${setActions(s, info.bodies)}
          ${Object.keys(s.materials || {}).length ? `<details class="sp-materials" data-materials="${escape(s.set)}" data-body="${escape(s.body)}"><summary>Materials</summary>
            <table><tbody>${materialRows(s.materials, info.materials)}</tbody></table>
            <button type="button" class="ghost small" data-save-materials>Save materials</button> <span class="meta" data-materials-result></span></details>` : ''}</li>`).join('')}</ul>`
      : '<p class="meta">No sets yet.</p>';
    const candidates = bodyCandidates(assets());
    el.querySelector('[data-register]').innerHTML = candidates.length
      ? `<div class="row"><select name="asset">${candidates.map((a) => `<option value="${escape(a.id)}">${escape(a.name)}</option>`).join('')}</select>
          <input type="text" name="bodyName" placeholder="body name, e.g. human-male"></div>
        <div class="row"><label>Dressed under its pieces <select name="bodyDressing"><option value="">Choose…</option>${DRESSINGS.map(([v, t]) => `<option value="${v}">${escape(t)}</option>`).join('')}</select></label>
          <input type="text" name="bodyDressingWords" placeholder="your own words, e.g. a dark linen tunic"></div>
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
        const { ok, body } = await post('/api/slot-parts/bodies', bodyPayload(asset, box.querySelector('[name=bodyName]').value, labels,
          box.querySelector('[name=bodyDressing]').value, box.querySelector('[name=bodyDressingWords]').value));
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
    if (!job.running && job.picks) { drawPicks(job, box); return; }
    const state = job.running ? 'Making' : job.returncode === 0 ? 'Made' : 'Stopped or failed';
    const what = { fit: ' (fit again)', refit: ' (carried to another body)', unreal: ' (Unreal export)' }[job.action] || '';
    box.innerHTML = `<h3>${escape(state)}: ${escape(job.set)} on ${escape(job.body)}${escape(what)}</h3>
      ${job.running ? '<button type="button" class="ghost small" data-cancel>Stop</button>' : ''}
      <div class="sp-log">${escape((job.log || []).join('\n'))}</div>`;
    box.querySelector('[data-cancel]')?.addEventListener('click', async () => { await post('/api/slot-parts/run/cancel'); refresh(); });
  }

  function drawPicks(job, box) {
    const outfit = Boolean(job.picks.concept);
    box.innerHTML = `<h3>${outfit ? 'Pick the outfit' : 'Pick the pictures'}: ${escape(job.set)} on ${escape(job.body)}</h3>
      <p class="meta">${outfit ? 'The outfit drawn on the body once per seed; the ticked one is cut into the pieces.'
        : 'The best-scored picture of each piece is ticked (a weapon or shield: the first); only the ticked ones go on to 3D.'}</p>
      ${pickRows(job.picks, info.base || '/output/')}
      <div class="row"><button type="button" class="primary" data-make-3d>${outfit ? 'Go on with this outfit' : 'Make the 3D'}</button></div><div data-picks-result></div>`;
    box.querySelector('[data-make-3d]').addEventListener('click', async () => {
      const picks = Object.fromEntries([...box.querySelectorAll('input[data-pick]:checked')].map((i) => [i.dataset.pick, Number(i.value)]));
      const { ok, body } = await post('/api/slot-parts/run', { resume: true, set: job.set, body: job.body, picks });
      box.querySelector('[data-picks-result]').innerHTML = ok ? '' : `<p class="sp-problems">${escape(body.error)}</p>`;
      if (ok) refresh();
    });
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
    else updateBodies();
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
  el.addEventListener('click', async (event) => {
    const save = event.target.closest('[data-save-materials]');
    if (save) {
      const box = save.closest('[data-materials]');
      const pieces = changedMaterials([...box.querySelectorAll('select[data-set-material]')]
        .map((m) => ({ slot: m.dataset.setMaterial, value: m.value, was: m.dataset.was })));
      const result = box.querySelector('[data-materials-result]');
      if (!Object.keys(pieces).length) { result.textContent = 'Nothing changed.'; return; }
      const { ok, body } = await post('/api/slot-parts/materials', { set: box.dataset.materials, body: box.dataset.body, pieces });
      result.textContent = ok ? 'Saved: the Unreal export uses these.' : body.error;
      if (ok) refresh();
      return;
    }
    const action = event.target.closest('[data-set-action]');
    if (action) {
      const row = action.closest('li');
      const payload = { action: action.dataset.setAction, set: action.dataset.set, body: action.dataset.body };
      if (payload.action === 'refit') payload.to = row.querySelector('[data-refit-to]')?.value;
      if (payload.action === 'unreal') payload.size_cm = Number(row.querySelector('[data-size-cm]')?.value || 180);
      const { ok, body } = await post('/api/slot-parts/run', payload);
      result.innerHTML = ok ? '' : `<p class="sp-problems">${escape(body.error)}</p>`;
      if (ok) refresh();
      return;
    }
    const button = event.target.closest('[data-preview]');
    if (button) { open(button.dataset.preview, button.dataset.title); close(); }
  });
  el.querySelector('[data-close]').focus();
  await refresh();
  drawForm();
}
