// Edge lab: draw lasso outlines (annotator.py) or load a map, then threshold and extract edges with the JS port of
// EdgeDetector.refine_edges (edge_engine.js). In the local app the Python function is run too and compared.
import { route, app, api, esc, $, $$, isStatic, slider, download } from './core.js';
import { stretch, polygonMask, refineEdges, edgeStats } from './edge_engine.js';
import { disc, blobs, exampleOutline, loadGrey } from './edge_sources.js';

const SOURCES = [['lasso', 'Draw a lasso'], ['disc', 'Test disc'], ['blobs', 'Smooth blobs'], ['map', 'Load a prediction map']];

route('edges', async () => {
  const st = { src: 'lasso', h: 144, w: 192, base: null, mask: null, pred: null, edges: null, thr: 0.5, width: 1,
    showMask: true, note: '', verts: null };
  app().innerHTML = `<p class="kicker">inference.py · EdgeDetector.refine_edges</p><h1>Edge lab</h1>
    <p class="lead">The network outputs a probability map of cell interior; edges are extracted from it afterwards. Here
      you supply the map: draw cells with the lasso (as in the annotation tool), use a synthetic test map, or load a
      prediction map saved from the Predict page. Then move the threshold and edge width.</p>
    <div class="seg" role="group" aria-label="Input">${SOURCES.map(([k, t]) => `<button data-src="${k}">${t}</button>`).join('')}</div>
    <div class="side" style="margin-top:14px"><div class="panel" id="ctl"></div>
      <div><div class="lab-stage" id="stage"><canvas id="cv" aria-label="Image with mask and detected edges"></canvas>
        <svg id="ov" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none" aria-hidden="true"></svg></div>
        <p class="stats" id="stats" aria-live="polite"></p><p class="small mute" id="note"></p></div></div>`;
  const cv = $('#cv'), ov = $('#ov'), ctx = cv.getContext('2d');

  function controls() {
    const c = $('#ctl');
    const lasso = st.src === 'lasso', map = st.src === 'map';
    c.innerHTML = `${lasso ? `<label for="img">Background image (optional, PNG/JPEG)</label><input type="file" id="img" accept="image/*">
      <button id="ex" class="alt">Add example outline</button><button id="clr" class="alt">Clear mask</button>` : ''}
      ${map ? `<label for="pm">Prediction map (greyscale PNG, white = 1)</label><input type="file" id="pm" accept="image/*">` : ''}
      <div id="sl"></div><label><input type="checkbox" id="sm" ${st.showMask ? 'checked' : ''}>Show pixels above threshold</label>
      <button id="dl" class="alt">Download edges (PNG)</button>`;
    const sl = $('#sl', c);
    slider(sl, { id: 'thr', label: 'Threshold', min: 0.05, max: 0.95, step: 0.01, value: st.thr, fmt: v => v.toFixed(2) },
      v => { st.thr = v; compute(); });
    slider(sl, { id: 'ew', label: 'Edge width (px)', min: 1, max: 7, step: 1, value: st.width }, v => { st.width = v; compute(); });
    $('#sm', c).addEventListener('change', e => { st.showMask = e.target.checked; draw(); });
    $('#dl', c).addEventListener('click', saveEdges);
    if (lasso) {
      $('#img', c).addEventListener('change', async e => { if (e.target.files[0]) await useImage(e.target.files[0], false); });
      $('#ex', c).addEventListener('click', () => addPolygon(exampleOutline(st.h, st.w)));
      $('#clr', c).addEventListener('click', () => { st.mask = new Uint8Array(st.h * st.w); st.pred = Float64Array.from(st.mask); compute(); });
    }
    if (map) $('#pm', c).addEventListener('change', async e => { if (e.target.files[0]) await useImage(e.target.files[0], true); });
  }

  function setSize(h, w) {
    st.h = h; st.w = w; cv.width = w; cv.height = h;
    ov.setAttribute('viewBox', `-0.5 -0.5 ${w} ${h}`);
  }

  function select(src) {
    st.src = src; st.verts = null;
    $$('[data-src]').forEach(b => { const on = b.dataset.src === src; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
    setSize(144, 192);
    st.base = null; st.mask = new Uint8Array(st.h * st.w);
    if (src === 'disc') { st.pred = disc(st.h, st.w); st.note = 'Synthetic: a disc of 1s on 0s, like the test in tests/test_inference.py.'; }
    else if (src === 'blobs') { st.pred = blobs(st.h, st.w); st.note = 'Synthetic: two Gaussian blobs (a smooth map), not a model output.'; }
    else if (src === 'map') { st.pred = new Float64Array(st.h * st.w); st.note = 'Load a prediction map PNG (for example one downloaded from the Predict page).'; }
    else { st.pred = new Float64Array(st.h * st.w); st.note = 'Drag on the image to draw a closed outline; each outline is filled into the mask, as in the annotation tool. Load your own image as a background if you like.'; }
    controls(); compute();
  }

  async function useImage(file, asMap) {
    try {
      const g = await loadGrey(file);
      setSize(g.h, g.w);
      st.mask = new Uint8Array(g.h * g.w);
      if (asMap) { st.pred = Float64Array.from(g.grey, v => v / 255); st.base = null; }
      else { st.base = stretch(g.grey); st.pred = new Float64Array(g.h * g.w); }
      st.note = `${esc(file.name)}: ${g.original[1]}×${g.original[0]} px${g.scaled ? `, shown and processed at ${g.w}×${g.h}` : ''}.
        ${asMap ? 'Grey levels / 255 are used as probabilities.' : 'Background contrast-stretched to the 1st–99th percentile (preprocess_for_visualization).'}`;
      compute();
    } catch (e) { st.note = `Could not read ${esc(file.name)}: ${esc(e.message)}`; draw(); }
  }

  function addPolygon(verts) {
    const m = polygonMask(verts, st.h, st.w);
    for (let i = 0; i < m.length; i++) if (m[i]) st.mask[i] = 1;
    st.pred = Float64Array.from(st.mask);
    compute();
  }

  let pyTimer = 0;
  function compute() {
    st.edges = refineEdges(st.pred, st.h, st.w, st.thr, st.width);
    draw();
    if (!isStatic()) { clearTimeout(pyTimer); pyTimer = setTimeout(checkPython, 250); }
  }

  async function checkPython() {
    const want = st.edges;
    try {
      const r = await api('/refine', { body: { height: st.h, width: st.w, pred: [...st.pred], threshold: st.thr, edge_width: st.width } });
      if (want !== st.edges) return;
      const py = new Uint8Array(st.h * st.w);
      r.edges.forEach(i => { py[i] = 1; });
      let diff = 0;
      for (let i = 0; i < py.length; i++) diff += py[i] !== want[i];
      $('#stats').insertAdjacentHTML('beforeend', diff ? ` · <span class="err">Python differs at ${diff} pixels</span>`
        : ` · <span class="ok">Python refine_edges: identical</span>`);
    } catch (e) { console.warn(e); }
  }

  function draw() {
    const { h, w } = st, img = ctx.createImageData(w, h), d = img.data;
    for (let i = 0; i < h * w; i++) {
      let g = st.base ? st.base[i] * 255 : (st.src === 'lasso' ? 18 : st.pred[i] * 200);
      let r = g, gg = g, b = g;
      if (st.showMask && st.pred[i] > st.thr) { r = r * 0.6; gg = gg * 0.6 + 242 * 0.4; b = b * 0.6 + 196 * 0.4; }
      if (st.edges[i]) { r = 255; gg = 40; b = 60; }  // red edges, as the repo's overlays
      d[4 * i] = r; d[4 * i + 1] = gg; d[4 * i + 2] = b; d[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    ov.innerHTML = st.verts && st.verts.length > 1
      ? `<polyline points="${st.verts.map(p => p.join(',')).join(' ')}" fill="none" stroke="#ff5470" stroke-width="${w / 300}"/>` : '';
    const s = edgeStats(st.pred, st.edges, st.thr);
    $('#stats').innerHTML = `${w}×${h} px · above threshold <b>${s.inside}</b> · edge pixels <b>${s.edge}</b>`;
    $('#note').innerHTML = st.note;
  }

  function saveEdges() {
    const c = document.createElement('canvas');
    c.width = st.w; c.height = st.h;
    const x = c.getContext('2d'), img = x.createImageData(st.w, st.h);
    st.edges.forEach((v, i) => { img.data.fill(v ? 255 : 0, 4 * i, 4 * i + 3); img.data[4 * i + 3] = 255; });
    x.putImageData(img, 0, 0);
    c.toBlob(b => download('edges.png', b));
  }

  // lasso drawing (pointer)
  const toGrid = e => {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width * st.w - 0.5, (e.clientY - r.top) / r.height * st.h - 0.5];
  };
  cv.addEventListener('pointerdown', e => { if (st.src !== 'lasso') return; cv.setPointerCapture(e.pointerId); st.verts = [toGrid(e)]; });
  cv.addEventListener('pointermove', e => { if (st.verts) { st.verts.push(toGrid(e)); draw(); } });
  const finish = () => { if (!st.verts) return; const v = st.verts; st.verts = null; if (v.length > 2) addPolygon(v); else draw(); };
  cv.addEventListener('pointerup', finish);
  cv.addEventListener('pointercancel', finish);

  $$('[data-src]').forEach(b => b.addEventListener('click', () => select(b.dataset.src)));
  select('lasso');
  addPolygon(exampleOutline(st.h, st.w));
});
