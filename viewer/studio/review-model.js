// The review queue's plain logic: no DOM, no three.js, so Node can test it as shipped.

export const STATE_LABELS = { to_review: 'To review', sent_back: 'Sent back', approved: 'Approved' };
export const STATUS_LABELS = { pass: 'Pass', warn: 'Warn', fail: 'Fail' };
/** The chip text for an item's status; a batch can queue an asset before its checks run. */
export const statusLabel = (status) => STATUS_LABELS[status] || 'Not checked';
export const VIEW_LABELS = { front: 'Front', side: 'Side', three_quarter: 'Three-quarter', top: 'Top' };

/** The queue split the way the sheet shows it: to review first, then sent back, then approved. */
export function queueSections(payload) {
  const items = payload?.items ?? [];
  return [
    { id: 'to_review', title: 'To review', items: items.filter((i) => i.state === 'to_review') },
    { id: 'sent_back', title: 'Sent back, waiting for a new version', items: items.filter((i) => i.state === 'sent_back') },
    { id: 'approved', title: 'Recently approved', items: payload?.approved ?? [] },
  ];
}

/** How many assets wait for a person: the number on the Review button. */
export function waitingCount(payload) {
  return (payload?.items ?? []).filter((i) => i.state === 'to_review').length;
}

/** "2 fail · 1 warn", "All checks pass", or "Not checked yet" (queued by a batch before its checks). */
export function summaryLine(item) {
  if (item?.status == null) return 'Not checked yet';
  const fail = item?.summary?.fail?.length ?? 0, warn = item?.summary?.warn?.length ?? 0;
  const parts = [fail && `${fail} fail`, warn && `${warn} warn`].filter(Boolean);
  return parts.length ? parts.join(' · ') : 'All checks pass';
}

/** The checks report as rows, failures first, then warnings, then passes; skipped last. */
export function checkRows(report) {
  const rank = { fail: 0, warn: 1, pass: 2 };
  const rows = [...(report?.results ?? [])]
    .sort((a, b) => rank[a.status] - rank[b.status] || a.check.localeCompare(b.check))
    .map((r) => ({ check: r.check, status: r.status, message: r.message }));
  const skipped = (report?.skipped ?? []).map((s) => ({ check: s.check, status: 'skip', message: s.reason }));
  return [...rows, ...skipped];
}

/** The silhouette overlays to show, in view order, with their scores. */
export function overlays(report) {
  const found = (report?.results ?? []).find((r) => r.check === 'silhouette');
  const urls = found?.details?.overlay_urls ?? {};
  return Object.keys(VIEW_LABELS).filter((v) => urls[v])
    .map((v) => ({ view: v, label: VIEW_LABELS[v], url: urls[v], score: found.details.views?.[v] }));
}

/** A send-back needs a note; an approval does not. */
export function canDecide(decision, note) {
  return decision === 'approved' || (decision === 'sent_back' && String(note ?? '').trim().length > 0);
}

/** The GLB a Library asset would be checked as: the furthest step it reached. */
export function checkableModel(asset) {
  return asset?.rigged || asset?.finished || asset?.model || null;
}

/** Library assets that have a model to check, as choices for the "Check an asset" form.
 *  An asset with Blender versions offers them first, newest first, as the Library opens on
 *  the newest; then its original. All share the asset's id: they are versions of one asset. */
export function checkChoices(assets) {
  return (assets ?? []).filter((a) => !a.hidden).flatMap((a) => {
    const base = { id: a.id, name: a.name, picture: a.picture ?? null };
    // A prop set's game-ready outputs are its props (what the Library shows), not the sheet;
    // each is its own asset in the queue.
    const props = (a.kind === 'prop set' ? a.props ?? [] : []).filter((p) => p?.file);
    if (props.length) {
      return props.map((p) => ({ ...base, id: propKey(a.id, p.name), name: `${a.name} · ${p.name}`, file: p.file, version: null }));
    }
    const versions = [...(a.blender_versions ?? [])].reverse()
      .filter((v) => v?.file).map((v) => ({ ...base, file: v.file, version: v.label || v.id }));
    const original = checkableModel(a) ? [{ ...base, file: checkableModel(a), version: versions.length ? 'original' : null }] : [];
    return [...versions, ...original];
  });
}

/** A prop's key in the queue: its set's id and its own name, kept to the server's key rules. */
export function propKey(setId, propName) {
  return `${setId}-${String(propName ?? 'prop').replace(/[^A-Za-z0-9_.-]+/g, '-')}`.slice(0, 120);
}

/** Whether the server will take an approval (see review.approval_blocker): the checks ran and
 *  the licence check did not fail. Sending back is always possible. */
export function canApprove(item) {
  if (item?.status == null) return false;
  return !(item?.summary?.fail ?? []).includes('license.components');
}

/** The request body for POST /api/review/checks. `reference` is a path under output/, or ''. */
export function checksRequest(choice, { reference = '', profile = 'prop', height = '' } = {}) {
  const body = { asset: choice.file, key: choice.id, profile };
  if (reference) body.reference = reference;
  const metres = Number(height);
  if (height !== '' && Number.isFinite(metres) && metres > 0) body.height = metres;
  return body;
}
