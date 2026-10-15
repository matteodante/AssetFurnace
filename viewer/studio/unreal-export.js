// Export for Unreal: the selected Library asset becomes files Unreal Engine 5.8 imports
// with no touch-ups (scripts/export_unreal.py), optionally checked in Unreal itself.
// The asset and file captured when Export is pressed decide what is exported, never the
// selection at response time.

const STYLE = `
.unreal-export { display: grid; gap: 12px; padding: 16px; border-bottom: 1px solid var(--line); }
.unreal-export h3 { margin: 0; font-size: 15px; }
.unreal-export p { margin: 0; }
.unreal-export button { justify-content: center; min-height: 40px; }
.unreal-export button:disabled { opacity: .5; cursor: default; }
.unreal-export .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.unreal-export input[type=number] { width: 100%; }
.unreal-export .check { display: flex; gap: 8px; align-items: center; font-size: 12.5px; color: var(--ash); }
.unreal-export [data-status]:empty, .unreal-export [data-source]:empty { display: none; }
.unreal-export [data-status].error { color: #f6d2b6; }
.unreal-export [data-source], .unreal-export [data-result] { overflow-wrap: anywhere; }
.unreal-export [data-result] { display: grid; gap: 8px; }
.unreal-export [data-result] ul { margin: 0; padding-left: 18px; font-size: 12.5px; }
.unreal-export [data-result] li.no { color: #f6d2b6; }
.unreal-export [data-result] img { width: 100%; border-radius: 8px; border: 1px solid var(--line); }
.unreal-export [data-result] a { text-underline-offset: 3px; }
`;

export const KIND_LABELS = { prop: 'Object', creature: 'Creature', equipment: 'Equipment piece' };

const basename = (path) => path?.split('/').pop() || '';

// What to export for an asset: a character exports its rig with its moves; anything
// else exports the model on screen (for a prop set, the prop being shown).
export function exportSource(asset, file) {
  if (!asset) return null;
  if (asset.kind === 'character' && asset.rigged) {
    return { file: asset.rigged, kind: 'creature', moves: (asset.clips || []).map((c) => c.file) };
  }
  const model = file || asset.finished || asset.model;
  return model ? { file: model, kind: 'prop', moves: [] } : null;
}

// Bodies an equipment piece can be worn by: every other rigged character.
export function bodiesFor(assets, asset) {
  return (assets || []).filter((a) => a.rigged && a.id !== asset?.id).map((a) => ({ name: a.name, file: a.rigged }));
}

export function requestBody({ source, kind, size, measure, body, moves, check }) {
  const data = { file: source.file, kind, size_cm: Number(size), measure, check: Boolean(check) };
  if (kind === 'creature' && moves) data.animations = source.moves;
  if (kind === 'equipment') data.body = body;
  return data;
}

export function sizeProblem(value) {
  const size = Number(value);
  if (!value || !Number.isFinite(size) || size < 0.1) return 'Give its real size in centimetres: a generated model has none of its own.';
  return null;
}

async function request(path, options = {}) {
  const response = await fetch(`/api/unreal-export${path}`, {
    ...options, headers: { 'X-AssetFurnace-Request': 'unreal-export', ...options.headers },
  });
  let data;
  try { data = await response.json(); }
  catch { throw new Error('Restart Studio to load Export for Unreal, then reload this page.'); }
  if (!response.ok) throw new Error(data.error || 'The export could not start. Try again.');
  return data;
}

export function createUnrealPanel({ root, url, assets }) {
  if (!document.getElementById('unreal-export-style')) {
    const style = document.createElement('style');
    style.id = 'unreal-export-style';
    style.textContent = STYLE;
    document.head.append(style);
  }
  root.innerHTML = `
    <h3>Export for Unreal</h3>
    <label class="field">What it is<select data-kind></select></label>
    <div class="pair">
      <label class="field">Real size (cm)<input data-size type="number" min="0.1" step="0.1" inputmode="decimal" placeholder="e.g. 180"></label>
      <label class="field">Measured as<select data-measure><option value="height">Height</option><option value="longest">Longest side</option></select></label>
    </div>
    <label class="field" data-body-label hidden>Worn by<select data-body></select></label>
    <label class="check" data-moves-label hidden><input type="checkbox" data-moves checked> Include its moves</label>
    <label class="check" data-check-label hidden><input type="checkbox" data-check> Check it in Unreal too (about two minutes)</label>
    <p class="hint" data-source></p>
    <button class="primary" data-export disabled>Export for Unreal</button>
    <p class="hint" data-status role="status" aria-live="polite"></p>
    <div data-result hidden></div>`;
  const el = (key) => root.querySelector(`[data-${key}]`);
  for (const [value, label] of Object.entries(KIND_LABELS)) el('kind').add(new Option(label, value));
  let context = null, capabilities = null, working = false, timer = null;

  function message(text, error = false) {
    el('status').textContent = text;
    el('status').classList.toggle('error', error);
  }
  function controls() {
    const kind = el('kind').value;
    el('body-label').hidden = kind !== 'equipment';
    el('moves-label').hidden = kind !== 'creature' || !context?.source.moves.length;
    el('check-label').hidden = !capabilities?.unreal;
    const blocked = !context || working || !capabilities?.blender || sizeProblem(el('size').value)
      || (kind === 'equipment' && !el('body').value);
    el('export').disabled = Boolean(blocked);
    for (const key of ['kind', 'size', 'measure', 'body', 'moves', 'check']) el(key).disabled = working;
  }
  async function load() {
    if (capabilities) return;
    try {
      capabilities = await request('');
      if (!capabilities.blender) message('Blender was not found; Export for Unreal needs it. Install it or set I2L_BLENDER.', true);
    } catch (error) { message(error.message, true); }
    controls();
  }
  function update(asset, file) {
    const source = exportSource(asset, file);
    if (context && !working && context.asset.id === asset?.id && context.source.file === source?.file) return;
    if (working) return;  // the running export keeps its own asset; the panel follows when it ends
    clearTimeout(timer);
    context = asset && source ? { asset, source } : null;
    root.hidden = !context;
    if (!context) return;
    el('kind').value = source.kind;
    el('measure').value = source.kind === 'prop' ? 'longest' : 'height';
    el('body').replaceChildren(new Option('Choose a character…', ''));
    for (const body of bodiesFor(assets(), asset)) el('body').add(new Option(body.name, body.file));
    el('source').textContent = `From ${basename(source.file)}${source.moves.length ? ` and ${source.moves.length} move(s)` : ''}`;
    el('result').hidden = true;
    message('');
    load();
    controls();
  }
  function render(job) {
    const result = el('result');
    result.replaceChildren();
    const list = document.createElement('ul');
    for (const file of job.files || []) {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = url(file.path);
      link.download = file.name;
      link.textContent = file.name;
      item.append(link);
      list.append(item);
    }
    const zip = document.createElement('a');
    zip.href = `/api/unreal-export/jobs/${job.id}/zip`;
    zip.textContent = 'Download all (.zip)';
    result.append(list, zip);
    for (const problem of job.problems || []) {
      const p = document.createElement('p');
      p.className = 'hint';
      p.textContent = `Problem: ${problem}`;
      result.append(p);
    }
    if (job.unreal) {
      const title = document.createElement('p');
      title.textContent = job.unreal.ok ? `Unreal ${job.unreal.engine || ''}: every check passed.` : 'Unreal: some checks failed.';
      const checks = document.createElement('ul');
      for (const r of job.unreal.results || []) {
        const li = document.createElement('li');
        li.className = r.ok ? '' : 'no';
        li.textContent = `${r.ok ? '✓' : '✗'} ${r.check}: ${r.detail}`;
        checks.append(li);
      }
      result.append(title, checks);
      const picture = (job.files || []).find((f) => f.name === 'unreal_check.png');
      if (picture) {
        const img = document.createElement('img');
        img.src = `${url(picture.path)}?t=${Date.now()}`;
        img.alt = 'The asset in Unreal, next to Manny playing the same idle';
        result.append(img);
      }
    }
    if (capabilities?.can_reveal) {
      const reveal = document.createElement('button');
      reveal.className = 'ghost';
      reveal.textContent = 'Show folder';
      reveal.addEventListener('click', async () => {
        try { await request(`/jobs/${job.id}/reveal`, { method: 'POST' }); }
        catch (error) { message(error.message, true); }
      });
      result.append(reveal);
    }
    result.hidden = false;
  }
  async function poll(id, ctx) {
    try {
      const job = await request(`/jobs/${id}`);
      if (job.state === 'done' || job.state === 'error') {
        working = false;
        if (context === ctx) {
          message(job.message, job.state === 'error');
          if (job.state === 'done') render(job);
        }
        controls();
        return;
      }
      if (context === ctx) message(job.message);
      timer = setTimeout(() => poll(id, ctx), 1500);
    } catch (error) {
      working = false;
      if (context === ctx) message(error.message, true);
      controls();
    }
  }
  el('kind').addEventListener('change', () => {
    el('measure').value = el('kind').value === 'prop' ? 'longest' : 'height';
    controls();
  });
  for (const key of ['size', 'body', 'measure']) el(key).addEventListener('input', controls);
  el('export').addEventListener('click', async () => {
    const ctx = context;
    if (!ctx || working) return;
    const problem = sizeProblem(el('size').value);
    if (problem) { message(problem, true); return; }
    working = true;
    controls();
    el('result').hidden = true;
    message('Starting the export…');
    try {
      const job = await request('/jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody({
          source: ctx.source, kind: el('kind').value, size: el('size').value, measure: el('measure').value,
          body: el('body').value, moves: el('moves').checked, check: el('check').checked && capabilities?.unreal,
        })),
      });
      await poll(job.id, ctx);
    } catch (error) {
      working = false;
      if (context === ctx) message(error.message, true);
      controls();
    }
  });
  return { update };
}
