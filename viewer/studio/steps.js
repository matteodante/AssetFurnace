// The Steps panel's actions: each step's settings, its start call, and its progress card.
// The job APIs live on the server; the studio sends them the asset's files. Prop sheets split here too.

import { followJob, plainError, statusUrlFor, jobEnd } from './jobs.js';
import { freshSeed, FACES } from './create.js';
import { familyPicker, familyNote } from './creature-family.js';
import { escapeHTML as esc } from '../components/html.js';

const fileName = (path) => path?.split('/').pop() ?? '';

export async function fileFrom(url, name) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not read ${name} (${response.status})`);
  return new File([await response.blob()], name);
}

export async function postStart(url, body) {
  const response = await fetch(url, { method: 'POST', body, ...(typeof body === 'string' ? { headers: { 'Content-Type': 'application/json' } } : {}) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

/**
 * The body for the step the asset can take next. `ctx` has engines, presets, url().
 * With `{ redo: true }`, the body for making a finished step again with other values: it
 * starts from what the step was made with (`asset.made`), and the steps before it stay.
 */
export function stepBody(stepId, asset, ctx, { redo = false } = {}) {
  const made = (redo && asset.made?.[stepId]) || {};
  if (stepId === 'model') {
    if (!asset.picture) return '<p class="hint">This asset has no picture on disk to build from.</p>';
    const [best, ...others] = ctx.engines;
    if (!best) return '<p class="hint">No 3D engine is installed yet. <a href="./setup.html">Install one in Setup</a>.</p>';
    const engines = [best, ...others];
    const seed = made.seed ?? 42;
    return `<p class="hint">${redo ? 'Makes a new 3D model from the same picture; the old one stays in the Library.' : 'Builds a textured 3D model from the picture.'} Takes a few minutes.</p>
      <label class="check"><input type="checkbox" data-k="newseed" checked><span>New seed <span class="hint">untick to choose it${redo && made.seed != null ? ` (it was ${esc(made.seed)})` : ''}: the same seed gives the same model</span></span></label>
      <label class="field" data-seedbox hidden>Seed<input type="number" data-k="seed" min="0" step="1" value="${esc(seed)}"></label>
      <details class="adv"><summary>Advanced</summary>
        <label class="field">Engine<select data-k="engine">${engines.map((e, i) => `<option value="${esc(e.id)}" ${made.engine === e.id ? 'selected' : ''}>${i ? '' : 'Best quality: '}${esc(e.label)}</option>`).join('')}</select></label>
        <p class="hint" data-k="licence">${esc((engines.find((e) => e.id === made.engine) || best).license)}</p>
      </details>
      <button class="primary" data-run="model">${redo ? 'Make it again' : 'Make 3D model'}</button>`;
  }
  if (stepId === 'finished' && asset.kind === 'prop set') {
    return `<p class="hint">Cuts the sheet into one model per prop, each cleaned up with lighter versions for far away. A few minutes.</p>
      <button class="primary" data-run="finished">Split into props</button>`;
  }
  if (stepId === 'finished') {
    if (!asset.picture) return '<p class="hint">Finishing needs the picture the model was made from, and this asset has none on disk.</p>';
    const faces = made.faces ?? 40000, texture = made.texture_size ?? 2048, photo = made.photo ?? true;
    return `${redo ? '<p class="hint">Finishes the current 3D model again; the old finish stays in the Library.</p>' : ''}
      <label class="field">Detail<select data-k="faces">
        ${facesChoices(faces).map(([v, t]) => `<option value="${v}"${v === faces ? ' selected' : ''}>${t}</option>`).join('')}</select></label>
      <label class="check"><input type="checkbox" data-k="photo" ${photo ? 'checked' : ''}><span>Pixel Match <span class="hint">keeps your picture's sharp details on the front</span></span></label>
      <details class="adv"><summary>Advanced</summary>
        <label class="field">Texture size<select data-k="texture">${[1024, 2048, 4096].map((v) => `<option value="${v}"${v === texture ? ' selected' : ''}>${v} px</option>`).join('')}</select></label>
      </details>
      <button class="primary" data-run="finished">${redo ? 'Finish again' : 'Finish model'}</button>`;
  }
  if (stepId === 'rigged') {
    return `<p class="hint">${redo ? 'Rigs the current model again; the old rig and its moves stay in the Library.' : 'Adds a skeleton so it can move.'} About a minute. Works best on a character standing in a T-pose with empty hands.</p>
      ${redo ? familyPicker('Rig again', asset.family_chosen ? asset.family : 'auto') : familyPicker()}`;
  }
  if (stepId === 'animated' && asset.fits_moves === false) {
    return familyNote(asset) || `<p class="hint">Rigged. Preset moves are for humanoids only for now: four-legged and other creatures can be rigged,
      not animated, and the rig may be wonky. Download it to animate in Blender.</p>`;
  }
  if (stepId === 'animated') return familyNote(asset) + movePicker(asset, ctx, asset.clips.length ? 'Add this move' : 'Animate');
  return '';
}

/** The triangle budgets to offer, with the one a step was made with when it is not one of them. */
export function facesChoices(faces) {
  if (FACES.some(([v]) => v === faces)) return FACES;
  return [...FACES, [faces, `${Number(faces).toLocaleString('en-US')} triangles · as made`]].sort((a, b) => a[0] - b[0]);
}

function movePicker(asset, ctx, cta) {
  const presets = ctx.presets.filter((p) => !p.take);
  if (!presets.length) return '<p class="hint">No preset moves are installed.</p>';
  const have = new Set(asset.clips.map((c) => c.name));
  return `<div class="moves" role="group" aria-label="Moves">${presets.map((p, i) => `<button class="chip" data-preset="${esc(p.id)}" aria-pressed="${i === 0}" ${have.has(p.id) ? 'data-tip="Already made: making it again replaces it"' : ''}>${esc(p.label)}${have.has(p.id) ? ' ✓' : ''}</button>`).join('')}</div>
    <details class="adv"><summary>Advanced</summary>
      <label class="field">Speed <output data-o="speed">1.00×</output><input type="range" data-k="speed" min="0.5" max="2" step="0.05" value="1"></label>
      <label class="field" data-tip="Bulky characters (armour, big shoulders) can hide their hands in their hips. This moves hanging arms outward a little; raised arms are left alone.">Arms away from body <output data-o="spread">0°</output><input type="range" data-k="spread" min="0" max="25" step="1" value="0"></label>
    </details>
    <button class="primary" data-run="animated">${cta}</button>`;
}

/** A step card's settings, read at once: the progress card replaces these controls as
 * soon as a step starts. */
export function readSettings(card, ctx) {
  const k = (key) => card.querySelector(`[data-k="${key}"]`);
  return {
    backend: k('engine')?.value || ctx.engines[0]?.id,
    faces: Number(k('faces')?.value), photo: k('photo')?.checked, texture: Number(k('texture')?.value),
    preset: card.querySelector('[data-preset][aria-pressed="true"]')?.dataset.preset,
    speed: Number(k('speed')?.value || 1), spread: Number(k('spread')?.value || 0), family: k('family')?.value,
    clips: k('clips')?.checked ?? true, variations: k('variations')?.checked ?? true,
  };
}

/** Wire a step card's controls. `ctx.onDone(asset)` reloads the Library afterwards. */
export function wireStep(card, stepId, asset, ctx) {
  const k = (key) => card.querySelector(`[data-k="${key}"]`);
  card.querySelectorAll('[data-preset]').forEach((chip) => chip.onclick = () => {
    card.querySelectorAll('[data-preset]').forEach((c) => c.setAttribute('aria-pressed', String(c === chip)));
  });
  const speed = k('speed'), spread = k('spread'), engine = k('engine'), newseed = k('newseed');
  if (newseed) newseed.onchange = () => { card.querySelector('[data-seedbox]').hidden = newseed.checked; };
  if (speed) speed.oninput = () => { card.querySelector('[data-o="speed"]').textContent = `${Number(speed.value).toFixed(2)}×`; };
  if (spread) spread.oninput = () => { card.querySelector('[data-o="spread"]').textContent = `${spread.value}°`; };
  if (engine) engine.onchange = () => { card.querySelector('[data-k="licence"]').textContent = ctx.engines.find((e) => e.id === engine.value)?.license || ''; };

  card.querySelectorAll('[data-run]').forEach((button) => button.onclick = async () => {
    const settings = readSettings(card, ctx);
    card.dataset.lastPreset = settings.preset || '';
    const seed = newseed && !newseed.checked ? Math.max(0, Math.trunc(Number(k('seed').value) || 0)) : freshSeed();
    const run = { model: () => startModel(asset, ctx, settings.backend, seed),
      finished: () => (asset.kind === 'prop set' ? startSplit(asset)
        : startFinish(asset, ctx, { faces: settings.faces, skip_photo: !settings.photo, texture_size: settings.texture })),
      rigged: () => startRig(asset, settings.family, { clips: settings.clips, variations: settings.variations }),
      animated: () => startAnimate(asset, settings.preset, settings.speed, settings.spread) }[stepId];
    await runInCard(card, run, ctx, stepId, '');
  });
}

async function startModel(asset, ctx, backend, seed = freshSeed()) {
  const form = new FormData();
  form.append('image', await fileFrom(ctx.url(asset.picture), fileName(asset.picture)));
  form.append('settings', JSON.stringify({ backend, seed }));  // a new try each time, unless a seed is kept
  return { kind: 'generate', start: await postStart('/api/generate', form) };
}

async function startFinish(asset, ctx, settings) {
  const model = asset.model || asset.finished;
  const form = new FormData();
  form.append('asset', await fileFrom(ctx.url(model), fileName(model)));
  form.append('image', await fileFrom(ctx.url(asset.picture), fileName(asset.picture)));
  form.append('settings', JSON.stringify(settings));
  return { kind: 'finish', start: await postStart('/api/finish', form) };
}

async function startSplit(asset) {
  const form = new FormData();
  form.append('generated', asset.model);
  form.append('settings', JSON.stringify({}));
  return { kind: 'props', start: await postStart('/api/props', form) };
}

async function startRig(asset, family = 'auto', { clips = true, variations = true } = {}) {
  const form = new FormData();
  form.append('model', asset.finished || asset.model);
  form.append('family', family || 'auto');
  form.append('clips', clips ? '1' : '0');
  form.append('variations', variations ? '1' : '0');
  return { kind: 'animate', start: await postStart('/api/animate/rig', form) };
}

async function startAnimate(asset, preset, speed, arm_spread) {
  if (!preset) throw new Error('Pick a move first.');
  return { kind: 'animate', start: await postStart('/api/animate/play', JSON.stringify({ model: asset.rigged, preset, speed, arm_spread })) };
}

const settingsOf = (card) => card.dataset.lastPreset || null;

async function runInCard(card, starter, ctx, stepId, note) {
  const body = card.querySelector('.body');
  const keep = body.innerHTML;
  const show = (html) => { body.innerHTML = html; };
  show(`<div class="job"><div class="bigbar"><span></span></div><p class="job-msg">Starting…</p>${note ? `<p class="hint">${esc(note)}</p>` : ''}
    <pre class="log" aria-label="Live log"></pre><div class="row end"><button class="ghost" data-cancel>Cancel</button></div></div>`);
  ctx.onBusy?.(true);
  let started;
  try {
    started = await starter();
  } catch (error) {
    ctx.onBusy?.(false);
    show(`<div class="job failed"><p class="job-msg">${esc(plainError(error.message))}</p><div class="row end"><button class="ghost" data-back>Back</button></div></div>`);
    body.querySelector('[data-back]').onclick = () => { body.innerHTML = keep; wireStep(card, stepId, ctx.asset(), ctx); };
    return;
  }
  const { kind, start } = started;
  const statusUrl = statusUrlFor(kind, start);
  const bar = body.querySelector('.bigbar span'), msg = body.querySelector('.job-msg'), log = body.querySelector('.log');
  const startedAt = Date.now();
  body.querySelector('[data-cancel]').onclick = async (event) => {
    event.target.disabled = true;
    await fetch(`/api/${kind}/${start.job_id}/cancel`, { method: 'POST' }).catch(() => {});
  };
  const job = followJob(statusUrl, {
    onUpdate: ({ event, percent, log: tail, stalled }) => {
      const minutes = Math.floor((Date.now() - startedAt) / 60000), seconds = Math.floor((Date.now() - startedAt) / 1000) % 60;
      msg.textContent = `${event.message || event.phase || 'Working…'} · ${minutes}:${String(seconds).padStart(2, '0')}`;
      if (percent != null) bar.style.width = `${percent}%`;
      else bar.classList.add('busy');
      log.textContent = tail.split('\n').slice(-5).join('\n');
      card.classList.toggle('stalled', stalled);
      if (stalled && !body.querySelector('.stallnote')) msg.insertAdjacentHTML('afterend', '<p class="stallnote">No news from this step for a few minutes. It may be stuck: you can keep waiting or cancel.</p>');
    },
  });
  const final = await job.done;
  ctx.onBusy?.(false);
  if (final.status === 'done') { ctx.onDone?.(stepId, { preset: stepId === 'animated' ? settingsOf(card) : null }); return; }
  const { cancelled, why, log: tail } = jobEnd(final);
  show(cancelled
    ? `<div class="job"><p class="job-msg">${esc(why)}</p><div class="row end"><button class="ghost" data-back>Back</button></div></div>`
    : `<div class="job failed"><p class="job-msg">${esc(why)}</p>
    <details><summary class="hint">Show log</summary><pre class="log">${esc(tail.split('\n').slice(-30).join('\n'))}</pre></details>
    <div class="row end"><button class="ghost" data-copy>Copy details</button><button class="ghost" data-back>Back</button></div></div>`);
  body.querySelector('[data-back]').onclick = () => { body.innerHTML = keep; wireStep(card, stepId, ctx.asset(), ctx); };
  body.querySelector('[data-copy]')?.addEventListener('click', () => navigator.clipboard?.writeText(`${why}\n\n${tail}`));
}
