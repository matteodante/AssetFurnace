// Animate from video: the Animate step for a rig the preset moves do not fit (four legs, wings, a tail).
// Four stages, each one run of scripts/twocam_mocap.py on the server (viewer/twocam_api.py): prepare the
// rig, teach the joint reader (once, about an hour), film takes, and use one as a move. What the card
// shows comes from the server, so a reopened page picks a running stage back up.

import { followJob, plainError, readableLog, copyDetails, TERMINAL } from './jobs.js';

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export const STAGES = [
  { id: 'prepare', label: 'Prepare' },
  { id: 'teach', label: 'Teach the reader' },
  { id: 'film', label: 'Film' },
  { id: 'capture', label: 'Capture' },
];
const VERDICT = { good: 'Good: both views agree', doubtful: 'Doubtful: the fit may be a little off', bad: 'Bad: the two views disagree' };
const TAKE_MINUTES = 6.5;   // one 3 s take at 1024 x 576 on an RTX 5070 Ti

/** "What is it?" starts as the asset's own title, without its version. */
export function defaultNoun(asset) {
  return String(asset?.name || '').replace(/\s*·\s*v\d+$/, '').trim();
}

/** The stage being worked on now, or null. */
export function runningJob(state) {
  const job = state?.job;
  return job && !TERMINAL.has(job.status) ? job : null;
}

/** done / now / later for each stage, from what is on disk. */
export function stageStates(state) {
  const prepared = !!state.prepared;
  const filmed = (state.moves || []).some((m) => m.takes.length);
  return {
    prepare: prepared ? 'done' : 'now',
    teach: !prepared ? 'later' : state.reader === 'taught' ? 'done' : 'now',
    film: !prepared ? 'later' : filmed ? 'done' : 'now',
    capture: !(prepared && state.reader !== 'none' && filmed) ? 'later' : state.captured?.length ? 'done' : 'now',
  };
}

function jobBox(job) {
  const event = job.last_event || {};
  const pct = Number(event.overall_pct);
  const away = ['teach', 'film'].includes(job.kind) ? '<p class="hint">It keeps going if you close this tab.</p>' : '';
  return `<div class="job" data-tc-job="${esc(job.job_id)}"><div class="bigbar"><span ${Number.isFinite(pct) ? `style="width:${pct}%"` : 'class="busy"'}></span></div>
    <p class="job-msg">${esc(event.message || 'Starting…')}</p>${away}
    <pre class="log">${esc(readableLog(job.log_tail, 3))}</pre>
    <div class="row end"><button class="ghost" data-tc-cancel>Cancel</button></div></div>`;
}

function failBox(job) {
  const event = job.last_event || {};
  const fitAnyway = event.bad_clip && event.take
    ? `<button class="ghost" data-tc-capture="${esc(event.take)}" data-force>Fit anyway</button>` : '';
  return `<div class="job failed" data-tc-failed="${esc(job.job_id)}"><p class="job-msg">${esc(plainError(event.message || job.log_tail || ''))}</p>
    <details><summary class="hint">Show log</summary><pre class="log">${esc(String(job.log_tail || '').split('\n').slice(-30).join('\n'))}</pre></details>
    <div class="row end">${fitAnyway}<button class="ghost" data-tc-copy>Copy details</button><button class="ghost" data-tc-dismiss>OK</button></div></div>`;
}

function prepareForm(state, asset, again) {
  const p = state.prepared || {};
  const startsOver = again && (state.reader !== 'none' || state.moves.length || state.poses?.done);
  return `<label class="field">How long is it, nose to tail, in metres?<input type="number" data-tc="length" min="0.05" max="100" step="0.05" value="${esc(p.length_m ?? '')}" placeholder="1.6"></label>
    <label class="field">What is it?<input type="text" data-tc="noun" maxlength="60" value="${esc(p.noun || defaultNoun(asset))}" placeholder="grey wolf"></label>
    <p class="hint">Turns it side-on, sizes it, names and paints its left and right limbs, and draws the picture every video starts from. About 10 seconds.</p>
    ${startsOver ? '<p class="hint">This starts the creature over: what the reader learned and the takes move to a folder called "earlier".</p>' : ''}
    <button class="primary" data-tc-act="prepare"${startsOver ? ' data-starts-over' : ''}>${again ? 'Prepare again' : 'Prepare'}</button>`;
}

function prepareBody(state, asset, url) {
  const p = state.prepared;
  if (!p) return prepareForm(state, asset, false);
  const preview = p.preview ? `<figure class="tc-preview"><img src="${esc(url(p.preview))}?v=${esc(p.updated)}" alt="The creature from the front and the side, its joints marked">
      <figcaption class="hint">Front on the left, side on the right. Its left joints are pink, its right ones blue. Check they sit on the right sides and it faces the way you expect.</figcaption></figure>` : '';
  return `${preview}${p.warning ? `<p class="stallnote">${esc(p.warning)}</p>` : ''}
    <details class="adv"><summary>Prepare again</summary>${prepareForm(state, asset, true)}</details>`;
}

function teachBody(state, st) {
  if (st === 'later') return '<p class="hint">After Prepare.</p>';
  if (st === 'done') return '';
  const dino = state.dino || {};
  const download = dino.cached ? '' : ` The first time, it downloads ${esc(dino.name)} (${esc(dino.megabytes)} MB, ${esc(dino.license)}), the network the reader starts from.`;
  const count = Number(state.poses?.count || 0).toLocaleString('en');
  const resume = state.reader === 'unfinished' || state.poses?.done
    ? `<p class="hint">A teaching stopped before the end. The ${Number(state.poses.done).toLocaleString('en')} practice poses already rendered are kept.</p>` : '';
  return `<p class="hint">Once per creature, about an hour. Renders ${count} practice poses, then teaches the reader to find this creature's joints in a video.${download}</p>
    ${resume}
    <button class="primary" data-tc-act="teach">${dino.cached ? 'Teach the reader' : `Download ${esc(dino.name)} (${esc(dino.megabytes)} MB) and teach the reader`}</button>`;
}

/** What filming needs before it can run: the engine, then the video model (its own Download button). */
function filmGate(state) {
  if (!state.engine_ready) {
    return { ok: false, html: '<p class="hint">Filming runs on stable-diffusion.cpp, which is not installed. Install Qwen-Image 2.1 on the <a href="./setup.html">Setup</a> page first.</p>' };
  }
  const plan = state.video_model || {};
  if (plan.ready) return { ok: true, html: '' };
  return { ok: false, html: `<div class="tc-plan"><p><b>The video model is not on this computer yet.</b> ${esc(plan.engine)}, about ${esc(plan.download_gb)} GB.</p>
      <p class="hint">Licence: <a href="${esc(plan.license_url)}" target="_blank" rel="noopener">${esc(plan.license)}</a>. ${esc(plan.license_note)}.</p>
      <button class="primary" data-tc-act="film" data-download>Download the video model (${esc(plan.download_gb)} GB) and film</button></div>` };
}

function takeHtml(take, state, url, canCapture) {
  const used = (state.captured || []).find((c) => c.take === take.clip);
  const level = take.check?.level;
  const verdict = level
    ? `<p class="tc-verdict ${esc(level)}">${esc(VERDICT[level] || level)}</p>${take.check.reasons?.length ? `<p class="hint">${take.check.reasons.map(esc).join('; ')}</p>` : ''}`
    : `<p class="hint">${state.reader === 'none' ? 'Judged once the reader is taught.' : 'Not judged.'}</p>`;
  const button = !canCapture ? ''
    : level === 'bad' ? `<button class="ghost small" data-tc-capture="${esc(take.clip)}" data-force>Fit anyway</button>`
      : `<button class="${used ? 'ghost' : 'primary'} small" data-tc-capture="${esc(take.clip)}">${used ? 'Use it again' : 'Use this take'}</button>`;
  return `<figure class="tc-take"><video src="${esc(url(take.clip))}" controls loop muted playsinline preload="metadata"></video>
    <figcaption><b>Take ${esc(take.number)}</b>${used ? ' · in use ✓' : ''}</figcaption>${verdict}${button}</figure>`;
}

function filmBody(state, st, url, canCapture) {
  if (st === 'later') return '<p class="hint">After Prepare.</p>';
  const gate = filmGate(state);
  const minutes = (n) => `${n} take${n > 1 ? 's' : ''} · about ${Math.round(n * TAKE_MINUTES)} min`;
  const form = `<label class="field">What does it do? (one sentence)<textarea data-tc="action" rows="2" maxlength="300" placeholder="It rears up and roars, then drops back onto all fours"></textarea></label>
    <label class="field">How does it end? (optional)<input type="text" data-tc="end" maxlength="200" placeholder="standing on all four legs"></label>
    <label class="field">Takes<select data-tc="takes">${[1, 2, 3].map((n) => `<option value="${n}"${n === 1 ? ' selected' : ''}>${minutes(n)}</option>`).join('')}</select></label>
    <p class="hint">MiniMax H3 films it front-on and side-on at once, on this computer.${state.reader === 'none' ? ' Takes filmed after the reader is taught are judged good, doubtful or bad.' : ' Each take is judged: good, doubtful or bad.'}</p>
    ${gate.ok ? '<button class="primary" data-tc-act="film">Film</button>' : gate.html}`;
  const filming = runningJob(state)?.kind === 'film' ? state.job.move : null;
  const moves = (state.moves || []).map((m) => `<div class="tc-move"><p class="tc-action">“${esc(m.action)}”${m.end ? ` <span class="hint">· ends ${esc(m.end)}</span>` : ''}</p>
      <div class="tc-takes">${m.takes.map((t) => takeHtml(t, state, url, canCapture)).join('') || `<p class="hint">${m.slug === filming ? 'Filming…' : 'No takes yet.'}</p>`}</div></div>`).join('');
  if (filming) return moves;   // the takes land one by one under the bar; the form waits
  return moves + (moves ? `<details class="adv"><summary>Film another move</summary>${form}</details>` : form);
}

function captureBody(state, st) {
  if (st === 'later') return `<p class="hint">${state.reader === 'none' ? 'Needs a taught reader and a filmed take.' : 'Needs a filmed take.'}</p>`;
  const made = (state.captured || []).map((c) => `<li>${esc(c.move.replace(/_/g, ' '))}${(c.warnings || []).length ? ` <span class="hint">· ${c.warnings.map(esc).join(' ')}</span>` : ''}</li>`).join('');
  return `<p class="hint">Pick a take above. "Use this take" reads its joints, fits the rig and adds the move to this model, in about a minute. Using another take of the same move replaces it.</p>
    ${made ? `<ul class="tc-made">${made}</ul>` : ''}`;
}

function stageSummary(stageId, state) {
  if (stageId === 'prepare' && state.prepared) return `${state.prepared.noun || ''} · ${state.prepared.length_m ?? '?'} m`;
  if (stageId === 'teach' && state.reader === 'taught') {
    const err = state.reader_info?.held_out_error_px;
    return err ? `Off by about ${((err.front + err.side) / 2).toFixed(1)} pixels on poses it had not seen` : 'Taught';
  }
  if (stageId === 'teach' && state.reader === 'unfinished') return 'Stopped before the end';
  if (stageId === 'capture' && state.captured?.length) return `${state.captured.length} move${state.captured.length > 1 ? 's' : ''} made`;
  return '';
}

/** The whole flow for one rig. `url` turns an output/ path into a served one; `dismissed` holds the
 *  failed jobs the person has already read. */
export function twocamHtml(state, asset, url, { dismissed = new Set() } = {}) {
  const states = stageStates(state);
  const job = state.job;
  const live = runningJob(state);
  const canCapture = state.reader !== 'none' && state.blender !== false && !live;
  const bodies = {
    prepare: () => prepareBody(state, asset, url),
    teach: (st) => teachBody(state, st),
    film: (st) => filmBody(state, st, url, canCapture),
    capture: (st) => captureBody(state, st),
  };
  const top = [
    state.blender === false ? '<p class="hint">Needs Blender, a free app, and it isn\'t installed. <a href="https://www.blender.org/download/" target="_blank" rel="noopener">Get Blender</a>, then reopen this page.</p>' : '',
    state.busy && !live ? `<p class="stallnote">${esc(plainError(state.busy))}</p>` : '',
  ].join('');
  const stages = STAGES.map((stage, i) => {
    const st = states[stage.id];
    const mine = job && job.kind === stage.id;
    const extra = mine && live ? jobBox(job)
      : mine && job.status === 'error' && !dismissed.has(job.job_id) ? failBox(job) : '';
    const sum = stageSummary(stage.id, state);
    return `<li class="tc-stage ${st}" data-stage="${stage.id}"><p class="tc-head"><span class="dot">${st === 'done' ? '✓' : i + 1}</span><b>${stage.label}</b>${sum ? `<span class="hint">${esc(sum)}</span>` : ''}</p>
      ${extra}${mine && live && stage.id !== 'film' ? '' : bodies[stage.id](st)}</li>`;
  }).join('');
  return `<p class="hint">Preset moves are for humanoids. This one moves from a video instead: an AI film of it from two cameras, read joint by joint and fitted to its rig.</p>
    ${top}<ol class="tc-stages">${stages}</ol>`;
}

async function call(path, body) {
  const response = await fetch(path, body === undefined ? undefined
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

// One card follows one job at a time: the Steps panel redraws often, and each redraw mounts it again.
let following = null;
const dismissed = new Set();
const lastState = new Map();

/**
 * Fill `box` (the Animate step's body) with the flow for `asset`'s rig and keep it live.
 * ctx: { url(rel), asset(), onDone(stepId, { preset }) }.
 */
export async function mountTwocam(box, asset, ctx) {
  following?.stop();
  following = null;
  const rig = asset.rigged;
  const render = (state) => {
    if (!box.isConnected) return;
    lastState.set(rig, state);
    box.innerHTML = twocamHtml(state, asset, ctx.url, { dismissed });
    wire(state);
  };
  const note = (message) => {
    box.querySelector('.tc-note')?.remove();
    box.insertAdjacentHTML('afterbegin', `<p class="stallnote tc-note">${esc(plainError(message))}</p>`);
  };
  const refresh = async () => {
    let state;
    try {
      state = await call(`/api/twocam/state?rig=${encodeURIComponent(rig)}`);
    } catch (error) { if (box.isConnected) note(error.message); return; }
    if (ctx.asset()?.rigged !== rig) return;   // the person moved on while this loaded
    render(state);
    const job = runningJob(state);
    if (job) follow(job);
  };

  function follow(job) {
    following?.stop();
    let takes = job.last_event?.takes_done ?? 0;
    const tracker = followJob(`/api/twocam/${job.job_id}/status`, {
      interval: 2000,
      onUpdate: ({ event, percent, log, stalled }) => {
        const el = box.querySelector(`[data-tc-job="${job.job_id}"]`);
        if (!box.isConnected || !el) { tracker.stop(); return; }
        const bar = el.querySelector('.bigbar span');
        if (percent != null) { bar.classList.remove('busy'); bar.style.width = `${percent}%`; } else bar.classList.add('busy');
        const eta = Number(event.stage_eta_seconds);
        el.querySelector('.job-msg').textContent = (event.message || 'Working…') + (Number.isFinite(eta) && eta > 60 ? ` · about ${Math.round(eta / 60)} min to go` : '');
        el.querySelector('.log').textContent = readableLog(log, 3);
        if (stalled && !el.querySelector('.stallnote')) el.querySelector('.job-msg').insertAdjacentHTML('afterend', '<p class="stallnote">No news from this step for a few minutes. It may be stuck: you can keep waiting or cancel.</p>');
        if (!stalled) el.querySelector('.stallnote')?.remove();
        if ((event.takes_done ?? 0) > takes) { takes = event.takes_done; refresh(); }   // a new take landed: show it
      },
    });
    following = tracker;
    tracker.done.then((final) => {
      if (following !== tracker || !box.isConnected) return;
      following = null;
      if (final.status === 'done' && job.kind === 'capture') { ctx.onDone?.('animated', { preset: final.last_event?.clip }); return; }
      refresh();
    });
  }

  function wire(state) {
    const live = runningJob(state);
    const start = async (button, kind, body) => {
      button.disabled = true;
      try {
        await call(`/api/twocam/${kind}`, { rig, ...body });
        await refresh();
      } catch (error) { note(error.message); button.disabled = false; }
    };
    box.querySelectorAll('[data-tc-act], [data-tc-capture]').forEach((button) => { if (live) button.disabled = true; });
    box.querySelectorAll('[data-tc-act]').forEach((button) => button.addEventListener('click', () => {
      const scope = button.closest('.tc-stage');
      const value = (key) => scope.querySelector(`[data-tc="${key}"]`)?.value.trim() ?? '';
      const kind = button.dataset.tcAct;
      if (kind === 'prepare') {
        const length = Number(value('length'));
        if (!(length > 0)) { note('Say how long it is, nose to tail, in metres: 1.6 for a wolf.'); return; }
        if (!value('noun')) { note('Say what it is, like "grey wolf": the video model is told.'); return; }
        if (button.dataset.startsOver !== undefined && !confirm('Prepare it again? What the reader learned and the takes move to the "earlier" folder, and the reader has to be taught again (about an hour).')) return;
        start(button, 'prepare', { length_m: length, noun: value('noun') });
      } else if (kind === 'teach') {
        start(button, 'teach', {});
      } else if (kind === 'film') {
        if (!value('action')) { note('Say what it does, in one sentence.'); return; }
        start(button, 'film', { action: value('action'), end: value('end') || undefined, takes: Number(value('takes') || 1),
          ...(button.dataset.download !== undefined ? { download: true } : {}) });
      }
    }));
    box.querySelectorAll('[data-tc-capture]').forEach((button) => button.addEventListener('click', () => {
      const force = button.dataset.force !== undefined;
      if (force && !confirm("This take's two views disagree, so the move will likely look wrong. Fit it anyway?")) return;
      start(button, 'capture', { take: button.dataset.tcCapture, ...(force ? { force: true } : {}) });
    }));
    box.querySelector('[data-tc-cancel]')?.addEventListener('click', async (event) => {
      if (live?.kind === 'teach' && !confirm('Stop teaching? The practice poses rendered so far are kept; the teaching itself starts over next time.')) return;
      event.target.disabled = true;
      await fetch(`/api/twocam/${live.job_id}/cancel`, { method: 'POST' }).catch(() => {});
    });
    const failed = state.job && box.querySelector(`[data-tc-failed="${state.job.job_id}"]`);
    failed?.querySelector('[data-tc-copy]')?.addEventListener('click', () => copyDetails(state.job.last_event?.message || '', state.job.log_tail || ''));
    failed?.querySelector('[data-tc-dismiss]')?.addEventListener('click', () => { dismissed.add(state.job.job_id); render(state); });
  }

  if (lastState.has(rig)) render(lastState.get(rig));
  else box.innerHTML = '<p class="hint">Loading…</p>';
  await refresh();
}
