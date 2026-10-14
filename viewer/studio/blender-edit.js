// Manual Blender exchange. The captured session, never the current selection at
// response time, decides which asset and source an imported file belongs to.

export const warningText = {
  rig_removed: 'This export has no skeleton. The source skeleton has not been copied over.',
  rig_changed: 'The rig or node transforms changed. Existing moves need checking before reuse.',
  animations_removed: 'This export contains fewer animations than its source.',
  textures_removed: 'This export contains fewer embedded textures than its source.',
  source_changed: 'The original file changed or was removed after this editing copy was created. The saved snapshot is still its parent.',
  provenance_unknown: 'No source license record was available. Editing does not establish usage rights.',
};

export function revisionFor(asset, id) {
  return (asset?.blender_versions || []).find((v) => v.id === id) || null;
}

export function initialRevision(asset, remembered) {
  const hasOriginal = asset?.model || asset?.finished || asset?.rigged || asset?.props?.length;
  if ((remembered === 'original' && hasOriginal) || revisionFor(asset, remembered)) return remembered;
  return asset?.blender_versions?.at(-1)?.id || 'original';
}

export function fileProblem(file, maximum = 512 * 1024 * 1024) {
  if (!file || !/\.glb$/i.test(file.name)) return 'Choose the glTF Binary (.glb) file exported from Blender.';
  if (!file.size || file.size > maximum) return 'Choose a nonempty GLB up to 512 MiB.';
  return null;
}

const basename = (path) => path?.split('/').pop() || '';
const dateLabel = (value) => new Date(value).toLocaleString();
// getRandomValues also works on a trusted LAN served over HTTP.
const requestId = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');

async function request(path, options = {}) {
  const response = await fetch(`/api/blender-edit${path}`, {
    ...options, headers: { 'X-AssetFurnace-Request': 'blender-edit', ...options.headers },
  });
  let data;
  try { data = await response.json(); }
  catch { throw new Error('Restart Studio to load Blender editing, then reload this page.'); }
  if (!response.ok) throw new Error(data.error || 'The edit could not be saved. Try again.');
  return data;
}

export function createBlenderPanel({ root, url, onVersion, onImported }) {
  root.innerHTML = `
    <h3>Edit in Blender</h3>
    <label class="field">Version<select data-version></select></label>
    <div data-details class="hint"></div>
    <label class="field" data-sessions-label hidden>Editing copy<select data-sessions></select></label>
    <p class="hint blender-source" data-source></p>
    <button class="ghost" data-open disabled>Open in Blender</button>
    <button class="ghost" data-download hidden>Download editing copy</button>
    <p class="hint" data-availability></p>
    <p class="hint">Edit in Blender, then File → Export → glTF 2.0. Choose <strong>glTF Binary (.glb)</strong> with textures and any animations you want to keep.</p>
    <button class="ghost" data-choose disabled>Choose exported GLB…</button>
    <input data-file type="file" accept=".glb,model/gltf-binary" hidden>
    <div data-pending hidden>
      <p class="hint" data-filename></p>
      <p class="hint">Only the contents of this export will be used. Old moves, LODs and other derived files stay with the original.</p>
      <div class="row"><button class="primary" data-import>Import as new version</button><button class="ghost" data-cancel>Cancel</button></div>
    </div>
    <p class="hint blender-status" data-status role="status" aria-live="polite"></p>`;
  const el = (key) => root.querySelector(`[data-${key}]`);
  let context = null, epoch = 0, working = false, selectedFile = null, timer = null;

  function message(text, error = false) {
    el('status').textContent = text;
    el('status').classList.toggle('error', error);
  }
  function controls() {
    el('open').disabled = working || !context?.capabilities?.can_open;
    el('choose').disabled = working || !context?.capabilities;
    el('download').disabled = working || !context?.capabilities;
    for (const key of ['version', 'sessions', 'import', 'cancel']) el(key).disabled = working;
  }
  function sourceLabel() {
    const source = context?.session?.parent_file || context?.file;
    el('source').textContent = `Source: ${basename(source)}`;
    el('source').title = source || '';
  }
  function details(version) {
    el('details').replaceChildren();
    if (!version) return;
    const p = document.createElement('p');
    p.textContent = `${dateLabel(version.created_at)} · ${version.facts.joints} joints · ${version.facts.animations.length} animations. From ${basename(version.parent_file)}.`;
    el('details').append(p);
    for (const code of version.warning_codes) {
      if (!warningText[code]) continue;
      const note = document.createElement('p');
      note.textContent = warningText[code];
      el('details').append(note);
    }
    for (const [label, path] of [['Metadata', version.metadata], ['Provenance', version.provenance]]) {
      const link = document.createElement('a');
      link.textContent = label;
      link.href = url(path);
      link.download = basename(path);
      el('details').append(link, document.createTextNode(' '));
    }
  }
  async function update(asset, file, version = null) {
    if (context && context.asset.id === asset?.id && context.file === file && context.version?.id === version?.id) return;
    const mine = ++epoch;
    clearTimeout(timer);
    selectedFile = null;
    el('file').value = '';
    el('pending').hidden = true;
    working = false;
    context = asset && file ? { asset, file, version, session: null, requestId: requestId() } : null;
    root.hidden = !context;
    if (!context) return;
    const ctx = context;
    el('version').replaceChildren(new Option('Original pipeline', 'original'));
    el('version').options[0].disabled = !(asset.model || asset.finished || asset.rigged || asset.props?.length);
    for (const [i, item] of (asset.blender_versions || []).entries()) {
      el('version').add(new Option(`Version ${i + 1} · ${item.label}`, item.id));
    }
    el('version').value = version?.id || 'original';
    details(version);
    el('sessions-label').hidden = true;
    el('availability').textContent = 'Checking Blender…';
    sourceLabel();
    message('');
    controls();
    try {
      ctx.capabilities = await request(`?asset_id=${encodeURIComponent(asset.id)}`);
      if (epoch !== mine) return;
      el('availability').textContent = ctx.capabilities.reason || 'Opens a separate editing copy on this computer. The original stays unchanged.';
      el('download').hidden = ctx.capabilities.can_open;
      const pending = ctx.capabilities.sessions;
      el('sessions').replaceChildren(new Option('New edit of the selected version', ''));
      for (const session of pending) el('sessions').add(new Option(`${basename(session.parent_file)} · ${dateLabel(session.created_at)}`, session.id));
      ctx.session = pending.find((s) => s.parent_file === file) || null;
      el('sessions').value = ctx.session?.id || '';
      el('sessions-label').hidden = !pending.length;
      sourceLabel();
      controls();
    } catch (error) {
      if (epoch !== mine) return;
      el('availability').textContent = '';
      message(`Blender editing is unavailable: ${error.message} Reload to retry.`, true);
    }
  }
  async function ensureSession(ctx) {
    if (!ctx.session) ctx.session = await request('/sessions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ asset_id: ctx.asset.id, file: ctx.file, request_id: ctx.requestId }),
    });
    return ctx.session;
  }
  async function poll(ctx, attempts = 0) {
    if (context !== ctx) return;
    try {
      const result = await request(`/sessions/${ctx.session.id}`);
      if (context !== ctx) return;
      if (result.launch.state === 'ready') { message('Blender is ready. Save your work there and export a GLB when finished.'); return; }
      if (result.launch.state === 'error') { message(result.launch.error, true); return; }
      if (attempts >= 30) { message('Blender is still opening. Check its window; you can import the exported GLB here when ready.'); return; }
      timer = setTimeout(() => poll(ctx, attempts + 1), 2000);
    } catch (error) { if (context === ctx) message(`Could not check Blender: ${error.message}`, true); }
  }
  el('version').addEventListener('change', () => onVersion(el('version').value));
  el('sessions').addEventListener('change', () => {
    context.session = context.capabilities.sessions.find((s) => s.id === el('sessions').value) || null;
    context.requestId = requestId();
    sourceLabel();
    message('');
  });
  el('open').addEventListener('click', async () => {
    const ctx = context;
    if (!ctx || working) return;
    working = true; controls(); message('Preparing an editing copy…');
    try {
      const session = await ensureSession(ctx);
      const result = await request(`/sessions/${session.id}/open`, { method: 'POST' });
      if (context !== ctx) return;
      sourceLabel();
      message(result.already_open ? 'This editing copy is already open in Blender.' : 'Opening Blender…');
      await poll(ctx);
    } catch (error) { if (context === ctx) message(error.message, true); }
    finally { if (context === ctx) { working = false; controls(); } }
  });
  el('choose').addEventListener('click', () => el('file').click());
  el('download').addEventListener('click', async () => {
    const ctx = context;
    if (!ctx || working) return;
    working = true; controls(); message('Preparing an editing copy…');
    try {
      const session = await ensureSession(ctx);
      if (context !== ctx) return;
      const link = document.createElement('a');
      link.href = `/api/blender-edit/sessions/${session.id}/source.glb`;
      link.download = 'blender-edit-source.glb';
      link.click();
      sourceLabel();
      message('Editing copy downloaded. Import it in Blender, then bring the exported GLB back here.');
    } catch (error) { if (context === ctx) message(error.message, true); }
    finally { if (context === ctx) { working = false; controls(); } }
  });
  el('file').addEventListener('change', () => {
    const file = el('file').files[0];
    if (!file) return;
    selectedFile = null;
    el('pending').hidden = true;
    const problem = fileProblem(file, context?.capabilities?.max_bytes);
    if (problem) { message(problem, true); el('file').value = ''; return; }
    selectedFile = file;
    el('filename').textContent = file.name;
    el('pending').hidden = false;
    message('');
    el('import').focus();
  });
  el('cancel').addEventListener('click', () => {
    selectedFile = null; el('file').value = ''; el('pending').hidden = true; el('choose').focus();
  });
  el('import').addEventListener('click', async () => {
    const ctx = context, file = selectedFile;
    if (!ctx || !file || working) return;
    working = true; controls(); message('Uploading and checking the exported GLB…');
    try {
      const session = await ensureSession(ctx);
      const version = await request(`/sessions/${session.id}/import`, {
        method: 'POST', headers: { 'Content-Type': 'model/gltf-binary', 'X-Filename': encodeURIComponent(file.name) }, body: file,
      });
      if (context === ctx) clearTimeout(timer);
      await onImported(ctx.asset.id, version.id, () => context === ctx);
    } catch (error) { if (context === ctx) message(error.message, true); }
    finally { if (context === ctx) { working = false; controls(); } }
  });
  function announceImported(assetId, versionId) {
    if (root.hidden || context?.asset.id !== assetId || context?.version?.id !== versionId) return;
    message('Imported as a new version. The original is unchanged.');
    el('version').focus();
  }
  return { update, announceImported };
}
