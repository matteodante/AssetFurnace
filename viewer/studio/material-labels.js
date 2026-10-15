// Material labels in the review panel: the user's three judgements on the version under
// review (painted light, materials look, material points clicked on the model), saved to
// /api/material-labels. They are the human set every material check is measured against.
// The pure functions come first and run under Node; three.js is only loaded to raycast.

import { escapeHTML } from '../components/html.js';

const HEADER = { 'X-AssetFurnace-Request': 'material-labels' };
const REPEAT = 1000, CLAMP = 1001, MIRROR = 1002;   // three.js wrapping constants

/** One texture coordinate wrapped the way the sampler wraps it. */
export function wrapUv(value, mode) {
  if (mode === CLAMP) return Math.min(1, Math.max(0, value));
  if (mode === MIRROR) {
    const period = value - 2 * Math.floor(value / 2);
    return period > 1 ? 2 - period : period;
  }
  return mode === REPEAT || mode == null ? value - Math.floor(value) : value;
}

/** A raycast hit as a material point: the texel the paint shows there (its own UV channel,
 *  through its texture transform and wrap, as three.js samples it), the glTF material index
 *  of the paint (never of a clay/normal look swapped in), and the word. */
export function pointFromHit(hit, label, indexOf) {
  const shown = hit?.object?.userData?.paint ?? hit?.object?.material;
  const material = Array.isArray(shown) ? shown[hit.face?.materialIndex ?? 0] : shown;
  const map = material?.map;
  const uv = map?.channel ? hit?.[`uv${map.channel}`] : hit?.uv;
  if (!uv) return null;
  let [u, v] = [uv.x, uv.y];
  const e = map?.matrix?.elements;
  if (e) [u, v] = [e[0] * u + e[3] * v + e[6], e[1] * u + e[4] * v + e[7]];
  const round = (x) => Math.round(x * 1e6) / 1e6;
  return { uv: [round(wrapUv(u, map?.wrapS ?? REPEAT)), round(wrapUv(v, map?.wrapT ?? REPEAT))],
           material: indexOf(material) ?? null, label };
}

export function isComplete(labels) {
  return labels?.baked_light != null && labels?.materials_look != null && (labels?.points?.length ?? 0) > 0;
}

/** Only items queued for labelling get the panel: labels on others would never be exported. */
export const inLabelSet = (item) => item?.source === 'material-labels';

/** Remove point `i` and the marker drawn for it (null for a point loaded from disk). */
export function dropAt(points, markers, i) {
  points.splice(i, 1);
  return markers.splice(i, 1)[0] ?? null;
}

/** A pointer that moved less than a few pixels between down and up is a click, not an orbit. */
export const isClick = (down, up) => Math.hypot(up.x - down.x, up.y - down.y) < 5;

/** A Yes/No-style button for one judgement; its value travels as JSON in an escaped attribute. */
export function choiceButton(field, value, text, current) {
  return `<button class="ghost small" data-ml-set="${escapeHTML(field)}" data-value="${escapeHTML(JSON.stringify(value))}" aria-pressed="${current === value}">${escapeHTML(text)}</button>`;
}

async function request(path, options = {}) {
  const response = await fetch(`/api/material-labels${path}`, { ...options, headers: { ...HEADER, ...options.headers } });
  let data;
  try { data = await response.json(); } catch { throw new Error('Restart Studio to load material labels, then reload this page.'); }
  if (!response.ok) throw new Error(data.error || 'The labels could not be saved.');
  return data;
}

// ------------------------------------------------------------------ panel

export function createLabelPanel() {
  let root = null, item = null, labels = null, vocabulary = [], view = null, render = null;
  let marking = false, viewer = null, down = null, epoch = 0, chosen = null, saving = false;
  const markers = [];

  const el = (key) => root?.querySelector(`[data-ml-${key}]`);
  const live = () => root?.isConnected && item;

  function draw() {
    if (!root || !labels) return;
    const typed = el('note');
    if (typed) labels.note = typed.value;   // a note typed but not saved survives the redraw
    const choice = (field, value, text) => choiceButton(field, JSON.parse(value), text, labels[field]);
    root.innerHTML = `
      <h3>Material labels</h3>
      <p class="meta" data-ml-progress></p>
      <div class="ml-row"><span>Light painted into the colours?</span>${choice('baked_light', 'true', 'Yes')}${choice('baked_light', 'false', 'No')}</div>
      <div class="ml-row"><span>Materials look</span>${choice('materials_look', '"right"', 'Right')}${choice('materials_look', '"wrong"', 'Wrong')}</div>
      <div class="ml-row"><span>Mark</span><select data-ml-label>${vocabulary.map((w) => `<option${w === chosen ? ' selected' : ''}>${escapeHTML(w)}</option>`).join('')}</select>
        <button class="ghost small" data-ml-mark aria-pressed="${marking}">${marking ? 'Stop marking' : 'Click the model'}</button></div>
      <ul class="ml-points">${labels.points.map((p, i) => `<li><span>${escapeHTML(p.label)}</span><span class="meta">uv ${p.uv.map((v) => v.toFixed(3)).join(', ')}</span><button class="ghost small" data-ml-drop="${i}" aria-label="Remove">×</button></li>`).join('') || '<li class="meta">No points yet: pick a material, then click where the model shows it.</li>'}</ul>
      <textarea data-ml-note rows="2" placeholder="Note (optional)">${escapeHTML(labels.note || '')}</textarea>
      <button class="primary" data-ml-save>Save labels</button>
      <p class="meta" data-ml-status aria-live="polite">${isComplete(labels) ? 'Complete.' : 'Give all three to complete this asset.'}</p>`;
  }

  async function progress() {
    try {
      const data = await request('');
      vocabulary = data.materials;
      const { complete, total } = data.progress;
      if (el('progress')) el('progress').textContent = `Label set: ${complete} of ${total} complete`;
    } catch { /* an older server: no labels */ }
  }

  function setMarking(on) {
    marking = on;
    if (on) viewer?.setTurntable?.(false);
    draw();
    progress();
  }

  async function show(container, reviewItem, options = {}) {
    const mine = ++epoch;
    if (!inLabelSet(reviewItem)) { item = null; return; }
    item = reviewItem;
    viewer = options.viewer || null;
    marking = false;
    root = document.createElement('li');
    root.className = 'ml-panel';
    container.appendChild(root);
    try {
      const [overview, current] = await Promise.all([request(''), request(`/item/${encodeURIComponent(item.key)}`)]);
      if (mine !== epoch) return;
      vocabulary = overview.materials;
      labels = current.labels;
      markers.length = 0;
      markers.push(...labels.points.map(() => null));   // points from disk have no marker
      draw();
      const { complete, total } = overview.progress;
      el('progress').textContent = `Label set: ${complete} of ${total} complete`;
    } catch (error) {
      if (mine === epoch) root.innerHTML = `<p class="meta">Material labels are not available: ${escapeHTML(error.message)}</p>`;
      return;
    }
    root.addEventListener('change', (event) => { if (event.target.matches('[data-ml-label]')) chosen = event.target.value; });
    root.addEventListener('click', async (event) => {
      const target = event.target.closest('button');
      if (!target || !labels) return;
      if (target.dataset.mlSet) {
        labels[target.dataset.mlSet] = JSON.parse(target.dataset.value);
        draw();
        progress();
      } else if (target.dataset.mlMark !== undefined) {
        setMarking(!marking);
      } else if (target.dataset.mlDrop !== undefined) {
        dropAt(labels.points, markers, Number(target.dataset.mlDrop))?.removeFromParent();
        render?.();
        draw();
        progress();
      } else if (target.dataset.mlSave !== undefined) {
        target.disabled = true;
        const savingFor = epoch;
        saving = true;
        root.inert = true;                       // no edits while the save is in flight: they would be lost
        try {
          const saved = await request(`/item/${encodeURIComponent(item.key)}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ asset_sha256: item.asset_sha256, baked_light: labels.baked_light,
              materials_look: labels.materials_look, points: labels.points, note: el('note').value }) });
          if (savingFor !== epoch) return;     // another item is open now: its panel is not ours to redraw
          labels = saved.labels;
          markers.length = labels.points.length;
          draw();
          await progress();
          el('status').textContent = saved.complete ? 'Saved. This asset is complete.' : 'Saved.';
        } catch (error) {
          if (savingFor !== epoch) return;
          el('status').textContent = `Not saved: ${error.message}`;
          target.disabled = false;
        } finally {
          saving = false;
          if (root) root.inert = false;
        }
      }
    });
  }

  async function pick(event) {
    if (saving || !marking || !live() || !view || !down || !isClick(down, { x: event.clientX, y: event.clientY })) return;
    const THREE = await import('three');
    const canvas = view.renderer.domElement;
    const box = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, view.camera);
    const hit = ray.intersectObject(view.modelRoot, true).find((h) => h.object.isMesh && h.uv);
    const indexOf = (material) => view.materials.find((entry) => entry.original === material)?.index ?? null;
    const point = hit && pointFromHit(hit, chosen || el('label').value, indexOf);
    if (!point) return;
    labels.points.push(point);
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.01, 12, 8), new THREE.MeshBasicMaterial({ color: 0xfd6d14, depthTest: false }));
    marker.renderOrder = 10;
    view.root.worldToLocal(marker.position.copy(hit.point));
    marker.scale.setScalar(1 / (view.root.scale.x || 1));
    view.root.add(marker);
    markers.push(marker);
    render?.();
    draw();
    progress();
  }

  return {
    show,
    /** A model loaded in the viewer: listen for marking clicks on it. */
    attach(loaded, requestRender) {
      view = loaded;
      render = requestRender;
      markers.fill(null);                     // the old model's markers went with it
      const canvas = loaded.renderer.domElement;
      canvas.addEventListener('pointerdown', (event) => { down = { x: event.clientX, y: event.clientY }; });
      canvas.addEventListener('pointerup', pick);
    },
  };
}
