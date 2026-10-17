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

// What to export for an asset: a selected Blender version exports that version (it holds
// only what was exported from Blender); a character exports its rig with its moves;
// anything else exports the model on screen (for a prop set, the prop being shown).
export function exportSource(asset, file, revision = null) {
  if (!asset) return null;
  if (revision?.file) {
    return { file: revision.file, kind: asset.kind === 'character' ? 'creature' : 'prop', moves: [] };
  }
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

// A size belongs to one model: another file (another prop of a set, another asset) starts
// empty, while a new move of the same rig keeps it.
export function keepsSize(previous, next) {
  return Boolean(previous && next) && previous.file === next.file;
}

// Whether two export sources are the same request: same file and the same moves.
export function sameSource(a, b) {
  return Boolean(a && b) && a.file === b.file && a.kind === b.kind
    && a.moves.length === b.moves.length && a.moves.every((move, i) => move === b.moves[i]);
}

// Equipment is scaled with the body that wears it, so its size is the wearer's.
export function sizeLabel(kind) {
  return kind === 'equipment' ? "Wearer's height (cm)" : 'Real size (cm)';
}

export function requestBody({ source, kind, size, measure, body, moves, check, material,
  name = '', collision = 'auto', lods = 'auto', tint = 'auto', forward = 'auto' }) {
  // Equipment is checked in Unreal together with its body's export, which the panel does
  // not have, so only from the command line.
  const data = { file: source.file, kind, size_cm: Number(size), measure, check: Boolean(check) && kind !== 'equipment' };
  if (kind === 'creature' && moves) data.animations = source.moves;
  if (kind === 'equipment') data.body = body;
  if (material) data.material = material; // empty: the rule picks it from the model's prompt
  // the export's own settings; "auto" (the default) is left to the export
  if (String(name).trim()) data.name = String(name).trim();
  if (kind === 'prop' && collision !== 'auto') data.collision = collision;
  if (kind === 'prop' && lods !== 'auto') data.lods = lods;
  if (tint !== 'auto') data.tint = tint;
  if (forward !== 'auto') data.forward = forward;
  return data;
}

// The material menu's own entries before the materials: Auto (the rule's) and none at all.
export const MATERIAL_CHOICES = [['', 'Auto'], ['none', 'Keep its own maps']];

export const COLLISIONS = [['auto', 'Auto (box)'], ['box', 'Box'], ['convex', 'Convex hull'], ['unreal', "Unreal's own"]];
export const FACINGS = [['auto', 'Auto'], ['-y', 'Front (-Y)'], ['y', 'Back (+Y)'], ['x', 'Right (+X)'], ['-x', 'Left (-X)']];

// "Auto: leather (the prompt says 'leather')": what the rule would pick, as the menu's first entry.
export function autoLabel(proposal) {
  return proposal?.material ? `Auto: ${proposal.material} (${proposal.reason})` : 'Auto';
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
      <label class="field"><span data-size-label>Real size (cm)</span><input data-size type="number" min="0.1" step="0.1" inputmode="decimal" placeholder="e.g. 180"></label>
      <label class="field">Measured as<select data-measure><option value="height">Height</option><option value="longest">Longest side</option></select></label>
    </div>
    <label class="field" data-body-label hidden>Worn by<select data-body></select></label>
    <label class="field" title="One material for the whole model: it sets roughness and metal in Unreal">Material<select data-material>${MATERIAL_CHOICES.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select></label>
    <label class="check" data-moves-label hidden><input type="checkbox" data-moves checked> Include its moves</label>
    <details class="adv" data-more><summary>More options</summary>
      <label class="field">Name in Unreal<input data-name type="text" maxlength="40" placeholder="from the file name"></label>
      <div class="pair" data-prop-only>
        <label class="field">Collision<select data-collision>${COLLISIONS.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select></label>
        <label class="field">Detail levels<select data-lods><option value="auto" title="the _LOD files beside the model">Auto</option><option value="none">None</option></select></label>
      </div>
      <div class="pair">
        <label class="field">Tint mask<select data-tint><option value="auto" title="the mask the material step left beside the model">Auto</option><option value="none">None</option></select></label>
        <label class="field" title="Which way the model faces in Blender; Auto reads a creature's family">Facing<select data-forward>${FACINGS.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select></label>
      </div>
    </details>
    <label class="check" data-check-label hidden><input type="checkbox" data-check> Check it in Unreal too (about two minutes)</label>
    <p class="hint" data-source></p>
    <button class="primary" data-export disabled>Export for Unreal</button>
    <p class="hint" data-status role="status" aria-live="polite"></p>
    <div data-result hidden></div>`;
  const el = (key) => root.querySelector(`[data-${key}]`);
  for (const [value, label] of Object.entries(KIND_LABELS)) el('kind').add(new Option(label, value));
  let context = null, capabilities = null, working = false, timer = null, pending = null;

  function message(text, error = false) {
    el('status').textContent = text;
    el('status').classList.toggle('error', error);
  }
  function controls() {
    const kind = el('kind').value;
    el('body-label').hidden = kind !== 'equipment';
    el('moves-label').hidden = kind !== 'creature' || !context?.source.moves.length;
    el('check-label').hidden = !capabilities?.unreal || kind === 'equipment';
    el('size-label').textContent = sizeLabel(kind);
    if (kind === 'equipment') el('measure').value = 'height';
    const blocked = !context || working || !capabilities?.blender || sizeProblem(el('size').value)
      || (kind === 'equipment' && !el('body').value);
    el('export').disabled = Boolean(blocked);
    for (const key of ['kind', 'size', 'measure', 'body', 'moves', 'check', 'material', 'name', 'collision', 'lods',
      'tint', 'forward']) el(key).disabled = working;
    el('prop-only').hidden = kind !== 'prop';
    el('measure').disabled = working || kind === 'equipment';
  }
  async function load() {
    if (capabilities) return;
    try {
      capabilities = await request('');
      for (const word of capabilities.materials || []) el('material').add(new Option(word, word));
      if (!capabilities.blender) message('Blender was not found; Export for Unreal needs it. Install it or set I2L_BLENDER.', true);
    } catch (error) { message(error.message, true); }
    controls();
  }
  function fillBodies(asset) {
    // Rebuilt on every update: other assets may have been rigged since.
    const chosen = el('body').value;
    el('body').replaceChildren(new Option('Choose a character…', ''));
    for (const body of bodiesFor(assets(), asset)) el('body').add(new Option(body.name, body.file));
    el('body').value = [...el('body').options].some((o) => o.value === chosen) ? chosen : '';
  }
  async function propose() {
    // The rule's pick for this model and kind, shown on "Auto"; a person may choose another.
    const ctx = context;
    if (!ctx) return;
    try {
      const proposal = await request(`/material?file=${encodeURIComponent(ctx.source.file)}&kind=${encodeURIComponent(el('kind').value)}`);
      if (context === ctx) el('material').options[0].textContent = autoLabel(proposal);
    } catch { el('material').options[0].textContent = 'Auto'; }
  }
  function update(asset, file, revision = null) {
    // The running export keeps its own asset; the latest selection is applied when it ends.
    if (working) { pending = [asset, file, revision]; return; }
    const source = exportSource(asset, file, revision);
    if (context && context.asset.id === asset?.id && sameSource(context.source, source)) {
      context.asset = asset;
      fillBodies(asset);
      controls();
      return;
    }
    clearTimeout(timer);
    // A generated model has no size of its own: another model never inherits this one's.
    if (context?.asset.id !== asset?.id || !keepsSize(context?.source, source)) el('size').value = '';
    context = asset && source ? { asset, source } : null;
    root.hidden = !context;
    if (!context) return;
    el('kind').value = source.kind;
    lastKind = source.kind;
    el('measure').value = source.kind === 'prop' ? 'longest' : 'height';
    fillBodies(asset);
    el('source').textContent = `From ${basename(source.file)}${source.moves.length ? ` and ${source.moves.length} move(s)` : ''}`;
    el('result').hidden = true;
    el('material').value = '';
    for (const key of ['collision', 'lods', 'tint', 'forward']) el(key).value = 'auto';
    el('name').value = '';
    message('');
    load();
    controls();
    propose();
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
  function finish() {
    working = false;
    controls();
    if (pending) {
      const next = pending;
      pending = null;
      update(...next);
    }
  }
  async function poll(id, ctx, failures = 0) {
    try {
      const job = await request(`/jobs/${id}`);
      if (job.state === 'done' || job.state === 'error') {
        if (context === ctx) {
          message(job.message, job.state === 'error');
          if (job.state === 'done') render(job);
        }
        finish();
        return;
      }
      if (context === ctx) message(job.message);
      timer = setTimeout(() => poll(id, ctx), 1500);
    } catch (error) {
      // A long Blender or Unreal run outlives a hiccup: keep the job and ask again.
      if (failures < 5) {
        if (context === ctx) message('Lost contact with Studio for a moment; still waiting for the export…');
        timer = setTimeout(() => poll(id, ctx, failures + 1), 3000);
        return;
      }
      if (context === ctx) {
        message(`${error.message} The export may still finish; its files land in output/unreal.`, true);
      }
      finish();
    }
  }
  let lastKind = null;
  el('kind').addEventListener('change', () => {
    const kind = el('kind').value;
    // An object's size is not a wearer's height, nor the other way round.
    if ((kind === 'equipment') !== (lastKind === 'equipment')) el('size').value = '';
    lastKind = kind;
    el('measure').value = kind === 'prop' ? 'longest' : 'height';
    controls();
    propose();
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
          material: el('material').value, name: el('name').value, collision: el('collision').value,
          lods: el('lods').value, tint: el('tint').value, forward: el('forward').value,
        })),
      });
      await poll(job.id, ctx);
    } catch (error) {
      if (context === ctx) message(error.message, true);
      finish();
    }
  });
  return { update };
}
