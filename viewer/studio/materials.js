// Game materials in the studio: run the step, read its checks, see the textures, and
// recolour the three tint zones live with the same formula Unreal uses:
//   out = albedo + m * g * (t - c),  g = luminance(albedo) / luminance(c),  in linear light.
// The pure functions come first and run under Node; three.js is only loaded by the panel.

import { escapeHTML } from '../components/html.js';

export const LUMINANCE = [0.2126, 0.7152, 0.0722];
const CHANNELS = ['R', 'G', 'B'];
const HEADER = { 'X-AssetFurnace-Request': 'materials' };

export const srgbToLinear = (v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
export const linearToSrgb = (v) => {
  const c = Math.min(1, Math.max(0, v));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
};
export const luminance = (rgb) => rgb.reduce((sum, v, i) => sum + v * LUMINANCE[i], 0);

export function hexToLinear(hex) {
  const value = /^#?([0-9a-f]{6})$/i.exec(hex)?.[1];
  if (!value) throw new Error(`not a #rrggbb colour: ${hex}`);
  return [0, 2, 4].map((i) => srgbToLinear(parseInt(value.slice(i, i + 2), 16) / 255));
}

export function linearToHex(rgb) {
  return '#' + rgb.map((v) => Math.round(linearToSrgb(v) * 255).toString(16).padStart(2, '0')).join('');
}

/** Shader constants per zone: scale = 1 / Y(c), delta = t - c. Untouched zones add nothing. */
export function tintUniforms(zones, tints) {
  const scale = [0, 0, 0], delta = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const zone of zones || []) {
    const i = CHANNELS.indexOf(zone.channel);
    const tint = tints?.[zone.channel];
    if (i < 0 || !tint) continue;
    const y = luminance(zone.mean_linear);
    scale[i] = y > 1e-6 ? 1 / y : 0;
    delta[i] = tint.map((t, k) => t - zone.mean_linear[k]);
  }
  return { scale, delta };
}

/** The formula on one linear pixel, exactly as the shader runs it. */
export function tintPixel(albedo, mask, { scale, delta }) {
  const y = luminance(albedo);
  return albedo.map((v, k) => {
    const shifted = v + [0, 1, 2].reduce((sum, i) => sum + mask[i] * y * scale[i] * delta[i][k], 0);
    return Math.min(1, Math.max(0, shifted));
  });
}

const DECLARE = `uniform sampler2D afTintMask;
uniform vec3 afScale;
uniform vec3 afDelta0;
uniform vec3 afDelta1;
uniform vec3 afDelta2;
uniform float afZones;
`;
const APPLY = `
#ifdef USE_MAP
  vec3 afMask = texture2D( afTintMask, vMapUv ).rgb;
  float afY = dot( diffuseColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
  vec3 afShift = afMask.r * afScale.x * afDelta0 + afMask.g * afScale.y * afDelta1 + afMask.b * afScale.z * afDelta2;
  diffuseColor.rgb = clamp( diffuseColor.rgb + afY * afShift, 0.0, 1.0 );
  diffuseColor.rgb = mix( diffuseColor.rgb, afMask, afZones * 0.7 * max( afMask.r, max( afMask.g, afMask.b ) ) );
#endif`;

/** Add the tint to a three.js standard material's fragment shader, once. */
export function patchShader(shader, uniforms) {
  Object.assign(shader.uniforms, uniforms);
  if (shader.fragmentShader.includes('afTintMask')) return shader;
  shader.fragmentShader = DECLARE + shader.fragmentShader.replace(
    '#include <map_fragment>', `#include <map_fragment>${APPLY}`);
  return shader;
}

const CHECKS = [
  ['tint_zones', 'Tint zones'],
  ['baked_light_direction', 'Painted light'],
  ['baked_light_highlights', 'Painted highlights'],
  ['baked_light_occlusion', 'Painted shadows in creases'],
];

/** The checks in a fixed order with plain labels. */
export function checkRows(checks) {
  return CHECKS.map(([id, label]) => {
    const found = (checks || []).find((c) => c.check === id);
    return found && { id, label, status: found.status, text: found.message };
  }).filter(Boolean);
}

/** The GLB whose paint the step reads: the finished model, else the raw one, else the rig. */
export function materialSource(asset, revision = null) {
  if (revision?.file) return revision.file;
  if (!asset) return null;
  if (asset.kind === 'prop set') return asset.props?.[0]?.file || asset.model || null;
  return asset.finished || asset.model || asset.rigged || null;
}

// ------------------------------------------------------------------ panel

const STATUS = { pass: 'passes', warn: 'look again', fail: 'sent back', skipped: 'not checked' };

export function createMaterialsPanel({ root, url }) {
  root.innerHTML = `
    <h3>Game materials</h3>
    <p class="hint" data-summary></p>
    <ul class="af-checks" data-checks></ul>
    <div class="af-zones" data-zones hidden>
      <div class="af-zone-row" data-pickers></div>
      <label class="af-toggle"><input type="checkbox" data-show-zones> Show zones</label>
      <button class="ghost small" data-reset>Original colours</button>
    </div>
    <div class="af-maps" data-maps hidden></div>
    <button class="ghost" data-run disabled>Make game materials</button>
    <p class="hint af-status" data-status role="status" aria-live="polite"></p>`;
  const el = (key) => root.querySelector(`[data-${key}]`);
  let file = null, run = null, view = null, render = null, uniforms = null, timer = null, epoch = 0;
  let tints = {};

  function message(text, error = false) {
    el('status').textContent = text;
    el('status').classList.toggle('error', error);
  }

  async function request(path, options = {}) {
    const response = await fetch(`/api/materials${path}`, { ...options, headers: { ...HEADER, ...options.headers } });
    let data;
    try { data = await response.json(); } catch { throw new Error('Restart Studio to load game materials, then reload this page.'); }
    if (!response.ok) throw new Error(data.error || 'The materials step could not start.');
    return data;
  }

  function renderRun() {
    const rows = checkRows(run?.checks);
    el('summary').textContent = !run ? 'Not made yet: a tint mask, normal and ORM textures, and a check for painted light.'
      : run.glb ? 'Made. Pick colours to recolour the three zones live.' : 'Sent back: see the checks below.';
    el('checks').innerHTML = rows.map((r) => `<li class="af-check ${r.status}"><b>${escapeHTML(r.label)}</b>
      <span class="af-pill">${STATUS[r.status] || escapeHTML(r.status)}</span><span class="meta">${escapeHTML(r.text)}</span></li>`).join('');
    const zones = run?.glb ? run.zones || [] : [];
    el('zones').hidden = !zones.length || !run?.files?.tint_mask;
    el('pickers').innerHTML = zones.map((z) => `<label class="af-zone" title="Zone ${z.channel}: ${Math.round(z.area_share * 100)}% of the surface">
      <input type="color" data-zone="${z.channel}" value="${linearToHex(tints[z.channel] || z.mean_linear)}">
      <span>${z.channel} · ${Math.round(z.area_share * 100)}%</span></label>`).join('');
    const maps = [['tint_mask', 'Tint mask'], ['normal', 'Normal'], ['orm', 'ORM']].filter(([key]) => run?.files?.[key]);
    el('maps').hidden = !maps.length;
    el('maps').innerHTML = maps.map(([key, label]) => `<a href="${escapeHTML(url(run.files[key]))}" target="_blank" rel="noopener">
      <img src="${escapeHTML(url(run.files[key]))}?v=${run.made}" alt="${label}" loading="lazy"><span>${label}</span></a>`).join('')
      + (run?.maps?.normal === 'kept' ? '<span class="af-kept">Normal: the model already had one, kept as it is</span>' : '');
    el('run').textContent = run ? 'Make again' : 'Make game materials';
  }

  async function applyTint() {
    if (!view || !run?.glb || !run.files?.tint_mask) return;
    const THREE = await import('three');
    const token = epoch;
    if (!uniforms) {
      uniforms = { afTintMask: { value: null }, afScale: { value: new THREE.Vector3() }, afZones: { value: 0 },
        afDelta0: { value: new THREE.Vector3() }, afDelta1: { value: new THREE.Vector3() }, afDelta2: { value: new THREE.Vector3() } };
    }
    const texture = await new THREE.TextureLoader().loadAsync(`${url(run.files.tint_mask)}?v=${run.made}`);
    if (token !== epoch) { texture.dispose(); return; }
    texture.flipY = false;                       // glTF UVs: v = 0 is the top row
    texture.colorSpace = THREE.NoColorSpace;     // the mask is data, not colour
    uniforms.afTintMask.value?.dispose();
    uniforms.afTintMask.value = texture;
    for (const entry of view.materials) {
      const material = entry.original;
      if (!material?.map || material.userData.afTint) continue;
      material.userData.afTint = true;
      material.onBeforeCompile = (shader) => patchShader(shader, uniforms);
      material.customProgramCacheKey = () => 'af-tint';
      material.needsUpdate = true;
    }
    updateUniforms();
  }

  function updateUniforms() {
    if (!uniforms || !run) return;
    const { scale, delta } = tintUniforms(run.zones, tints);
    uniforms.afScale.value.set(...scale);
    delta.forEach((d, i) => uniforms[`afDelta${i}`].value.set(...d));
    uniforms.afZones.value = el('show-zones').checked ? 1 : 0;
    render?.();
  }

  async function refresh() {
    clearTimeout(timer);
    const mine = ++epoch;
    if (!file) return;
    try {
      const data = await request(`?file=${encodeURIComponent(file)}`);
      if (mine !== epoch) return;
      run = data.run;
      renderRun();
      const { state, error } = data.job;
      el('run').disabled = state === 'running';
      if (state === 'running') {
        message('Making materials… this takes from a few seconds to a minute.');
        timer = setTimeout(refresh, 1500);
      } else {
        message(state === 'failed' ? `The step stopped: ${error}` : state === 'sent_back' ? 'Sent back: the albedo needs fixing first.' : '', state === 'failed');
        await applyTint();
      }
    } catch (error) {
      if (mine === epoch) message(error.message, true);
    }
  }

  el('run').addEventListener('click', async () => {
    if (!file) return;
    el('run').disabled = true;
    try {
      await request('/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file }) });
      refresh();
    } catch (error) {
      message(error.message, true);
      el('run').disabled = false;
    }
  });
  el('pickers').addEventListener('input', (event) => {
    const channel = event.target.dataset.zone;
    if (!channel) return;
    tints[channel] = hexToLinear(event.target.value);
    updateUniforms();
  });
  el('show-zones').addEventListener('change', updateUniforms);
  el('reset').addEventListener('click', () => { tints = {}; renderRun(); updateUniforms(); });

  return {
    /** Follow the asset on show; the same source keeps its picked colours. */
    update(asset, revision = null) {
      const next = materialSource(asset, revision);
      root.hidden = !next || !/\.glb$/i.test(next);
      if (next === file) return;
      file = next;
      run = null;
      tints = {};
      el('show-zones').checked = false;
      message('');
      renderRun();
      el('run').disabled = !file;
      if (file) refresh();
    },
    /** A model just loaded in the viewer: recolour its materials if this asset has a mask. */
    attach(loaded, requestRender) {
      view = loaded;
      render = requestRender;
      applyTint();
    },
  };
}
