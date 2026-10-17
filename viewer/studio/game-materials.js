// Game materials: the materials step on the model shown (viewer/materials_api.py). Its own
// maps packed as a game ORM; tint zones only when asked, found by a segmenter or given as a
// file of the person's own. No zone painting here. The pure parts first, for Node tests.

import { exportSource } from './unreal-export.js';

const escape = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** The run request from the panel's state; throws with what is missing, in words. */
export function requestFor({ file, zones = 'off', materials = '', zonesFile = null, zonesMask = null }) {
  const data = { file, zones };
  if (zones === 'file') {
    if (!zonesFile) throw new Error("Choose your zones file: a JSON of zones, or an earlier run's .materials.json.");
    if (!zonesMask) throw new Error("Choose the zones mask: the PNG of the zones in UV space, or that run's _Tint.png.");
    data.zones_file = zonesFile;
    data.zones_mask = zonesMask;
  } else if (zones !== 'off' && String(materials).trim()) {
    data.materials = String(materials).trim();
  }
  return data;
}

async function request(path, options = {}) {
  const response = await fetch(`/api/materials${path}`, {
    ...options, headers: { 'X-AssetFurnace-Request': 'materials', ...options.headers },
  });
  let data;
  try { data = await response.json(); } catch { throw new Error('Restart Studio to load Game materials, then reload this page.'); }
  if (!response.ok) throw new Error(data.error || 'The run could not start. Try again.');
  return data;
}

async function readText(file) {
  return { name: file.name, text: await file.text() };
}

async function readData(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return { name: file.name, data: btoa(binary) };
}

export function createMaterialsPanel({ root, url, onMade = () => {} }) {
  root.innerHTML = `
    <h3>Game materials</h3>
    <p class="hint">Packs the model's own roughness, metal and occlusion as a game ORM. Tint zones (the colours a game can recolour) are optional.</p>
    <label class="field">Tint zones<select data-zones>
      <option value="off">None: pack the maps only</option>
      <option value="sam31">Find them (SAM 3.1, uses the GPU)</option>
      <option value="file">From my own file…</option></select></label>
    <label class="field" data-words-label hidden>Materials<input data-words type="text" placeholder="optional, e.g. leather, metal, cloth"></label>
    <div data-own hidden>
      <label class="field">Zones file (.json)<input data-zones-file type="file" accept=".json,application/json"></label>
      <label class="field">Zones mask (.png)<input data-zones-mask type="file" accept=".png,image/png"></label>
      <p class="hint">A JSON like {"zones": {"1": "metal", "2": "leather"}} with a PNG of those zone numbers in UV space, or an earlier run's .materials.json corrected by hand with its _Tint.png.</p>
    </div>
    <button class="primary" data-run disabled>Make game materials</button>
    <p class="hint" data-status role="status" aria-live="polite"></p>
    <ul class="hint" data-runs></ul>`;
  const el = (key) => root.querySelector(`[data-${key}]`);
  let source = null, working = false;

  function message(text, error = false) {
    el('status').textContent = text;
    el('status').classList.toggle('error', error);
  }
  function controls() {
    const zones = el('zones').value;
    el('words-label').hidden = zones === 'off' || zones === 'file';
    el('own').hidden = zones !== 'file';
    el('run').disabled = !source || working;
  }
  async function listRuns() {
    const file = source?.file;
    if (!file) return;
    try {
      const { runs } = await request(`?file=${encodeURIComponent(file)}`);
      if (source?.file !== file) return;
      el('runs').innerHTML = runs.map((run) => `<li>${run.zones.length ? escape(run.zones.join(', ')) : 'maps packed, no zones'}
        · <a href="${escape(url(run.glb))}" download>GLB</a>${run.tint ? ` · <a href="${escape(url(run.tint))}" download>tint mask</a>` : ''}</li>`).join('');
    } catch { el('runs').innerHTML = ''; }
  }
  async function poll(id) {
    for (;;) {
      const job = await request(`/jobs/${id}`);
      if (job.status !== 'running') return job;
      message(job.message || 'Working…');
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  }
  el('zones').addEventListener('change', controls);
  el('run').addEventListener('click', async () => {
    if (!source || working) return;
    try {
      const zones = el('zones').value;
      const chosen = (key) => el(key).files?.[0] || null;
      const body = requestFor({
        file: source.file, zones, materials: el('words').value,
        zonesFile: zones === 'file' && chosen('zones-file') ? await readText(chosen('zones-file')) : null,
        zonesMask: zones === 'file' && chosen('zones-mask') ? await readData(chosen('zones-mask')) : null,
      });
      working = true;
      controls();
      message('Starting…');
      const job = await poll((await request('/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).id);
      if (job.status === 'done') {
        message(job.run?.tint ? 'Done: Export for Unreal can use its tint mask.' : 'Done.');
        onMade(job.run);
      } else message(job.error || 'The run failed.', true);
    } catch (error) {
      message(error.message, true);
    } finally {
      working = false;
      controls();
      listRuns();
    }
  });

  function update(asset, file, revision = null) {
    const next = exportSource(asset, file, revision);
    root.hidden = !next;
    if (next?.file !== source?.file) {
      el('runs').innerHTML = '';
      if (!working) message('');
    }
    source = next;
    controls();
    listRuns();
  }
  return { update };
}
