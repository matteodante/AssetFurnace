// Creature families in the Steps panel: the family picker sent with a rig, and what the
// Animate step says about a rig that is not a plain humanoid. The server side is
// viewer/creature_family_api.py; the families come from image_to_3dlab/creature_family.py.

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export const FAMILIES = [
  ['auto', 'Recognise it for me'],
  ['humanoid', 'Humanoid'],
  ['digitigrade', 'Two legs, digitigrade (werewolf, demon)'],
  ['quadruped', 'Four legs'],
  ['multileg', 'Many legs (spider, insect)'],
  ['winged', 'Winged (dragon, bird)'],
  ['serpent', 'Serpent'],
  ['plant', 'Plant'],
];

/** The Rig step's controls: what kind of creature this is, then one button. */
export function familyPicker(label = 'Rig it', chosen = 'auto') {
  return `<label class="field" data-tip="Picks which test clips the rig gets. Recognising it reads the skeleton: legs, arms, wings, tail.">What is it?
      <select data-k="family">${FAMILIES.map(([id, name]) => `<option value="${id}"${id === chosen ? ' selected' : ''}>${esc(name)}</option>`).join('')}</select></label>
    <div class="row"><button class="primary" data-run="rigged">${esc(label)}</button></div>`;
}

/** A line for the Animate step about the rig's family and its test clips, or ''. */
export function familyNote(asset) {
  if (!asset.family || asset.family === 'humanoid') return '';
  const library = new Set(asset.library_clips || []);
  const own = (asset.clips || []).filter((c) => c.name.startsWith(`${asset.family}-`)).map((c) => c.name);
  const fromLibrary = own.filter((name) => library.has(name));
  const tests = own.filter((name) => !library.has(name));
  const label = esc(asset.family_label || asset.family);
  const how = asset.family_chosen ? 'as you chose' : 'recognised from its skeleton';
  if (own.length) {
    const parts = [];
    if (fromLibrary.length) parts.push(`Clips from the CC0 creature library: ${fromLibrary.map(esc).join(', ')}.`);
    if (tests.length) parts.push(`Test clips to check the rig: ${tests.map(esc).join(', ')}.`);
    return `<p class="hint">Rigged as <b>${label}</b> (${how}). ${parts.join(' ')} Pick one under the model to play it.</p>`;
  }
  const why = asset.clips_error ? ` The test clips failed: ${esc(asset.clips_error)}` : ' This family has no test clips yet.';
  return `<p class="hint">Rigged as <b>${label}</b> (${how}).${why}</p>`;
}
