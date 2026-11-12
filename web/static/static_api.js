// Static-site data layer: answers api() calls without the Python server.
// GET paths come from the JSON exported by scripts/build_static.py. The edge lab computes in the browser
// (edge_engine.js) and does not call the API; previews and predictions need the local app.
const files = new Map();
function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'models') return { tensorflow: false, models: [], folder: 'models/', static: true };
  if (['preview', 'predict', 'refine'].includes(clean)) throw new Error('This needs the local app (Python server).');
  return file(clean);
}
