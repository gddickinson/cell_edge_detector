// Predict page (local app): upload a TIRF and a DIC TIFF stack, pick a trained model from models/, run
// EdgeDetector.detect_edges + refine_edges in Python. The static site explains what is needed instead.
import { route, app, api, esc, $, isStatic, needsServer, download } from './core.js';

const img = (b64, alt) => `<img src="data:image/png;base64,${b64}" alt="${esc(alt)}">`;

route('predict', async () => {
  const head = `<p class="kicker">inference.py · EdgeDetector</p><h1>Predict cell edges</h1>
    <p class="lead">Run a trained U-Net on one TIRF/DIC frame pair: the probability map, then the edges from
      <code>refine_edges</code> drawn in red on both images, as <code>batch_process</code> saves them.</p>`;
  if (isStatic()) {
    app().innerHTML = `${head}${needsServer('Prediction')}
      <div class="callout">The repository has no trained weights (the trained network would hold about 124 MB of float32
        weights), so nothing can be predicted in the browser. Train a model with the Python workflow (see the
        <a href="#/pipeline">pipeline</a>): it is saved as <code>models/unet_training_&lt;timestamp&gt;/best_model.h5</code>.
        Then start the local app (<code>python run_web.py</code>) and this page lists it.</div>
      <p>Meanwhile the <a href="#/edges">edge lab</a> runs the edge-extraction step in your browser on masks you draw.</p>`;
    return;
  }
  const m = await api('/models', { fresh: true });
  const ready = m.tensorflow && m.models.length;
  app().innerHTML = `${head}
    ${!m.tensorflow ? '<p class="static-note">TensorFlow is not installed in this Python environment: pip install tensorflow (see requirements.txt).</p>' : ''}
    ${!m.models.length ? `<p class="static-note">No trained model found under ${esc(m.folder)}. Train one first (see the <a href="#/pipeline">pipeline</a>).</p>` : ''}
    <form class="panel" id="f">
      <div class="row"><div><label for="model">Trained model</label><select id="model" name="model">${m.models.map(x =>
        `<option value="${esc(x.name)}">${esc(x.name)} (${x.mb} MB)</option>`).join('') || '<option value="">none found</option>'}</select></div>
        <div><label for="frame">Frame</label><input id="frame" name="frame" type="number" min="0" value="0"></div></div>
      <div class="row"><div><label for="tirf">TIRF stack (.tif)</label><input id="tirf" name="tirf" type="file" accept=".tif,.tiff"></div>
        <div><label for="dic">DIC stack (.tif)</label><input id="dic" name="dic" type="file" accept=".tif,.tiff"></div></div>
      <div class="row"><div><label for="threshold">Threshold</label><input id="threshold" name="threshold" type="number" min="0.01" max="0.99" step="0.01" value="0.5"></div>
        <div><label for="edge_width">Edge width (px)</label><input id="edge_width" name="edge_width" type="number" min="1" max="15" value="1"></div></div>
      <button id="go" ${ready ? '' : 'disabled'}>Predict</button></form>
    <div class="figs" id="prev"></div><p id="msg" aria-live="polite"></p><div id="out"></div>`;
  const prev = {};
  for (const id of ['tirf', 'dic']) {
    $('#' + id).addEventListener('change', async e => {
      const file = e.target.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append('file', file); fd.append('frame', $('#frame').value);
      try {
        const p = await api('/preview', { body: fd });
        prev[id] = `<figure>${img(p.png, `${id.toUpperCase()} frame ${p.frame}`)}<figcaption>${id.toUpperCase()}:
          ${esc(p.name)}, ${p.frames} frame(s), ${p.width}×${p.height} ${esc(p.dtype)} (frame ${p.frame}, stretched)</figcaption></figure>`;
      } catch (err) { prev[id] = `<p class="err">${esc(err.message)}</p>`; }
      $('#prev').innerHTML = (prev.tirf || '') + (prev.dic || '');
    });
  }
  $('#f').addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData($('#f'));
    if (!$('#tirf').files[0] || !$('#dic').files[0]) { $('#msg').innerHTML = '<span class="err">Choose a TIRF and a DIC file.</span>'; return; }
    $('#go').disabled = true; $('#msg').textContent = 'Running the U-Net...'; $('#out').innerHTML = '';
    try {
      const r = await api('/predict', { body: fd });
      $('#msg').innerHTML = `Frame ${r.frame}, ${r.width}×${r.height} px, model input ${r.model_input.join('×')}:
        ${(100 * r.fraction_above_threshold).toFixed(1)}% of pixels above ${r.threshold}, ${r.edge_pixels} edge pixels.`;
      $('#out').innerHTML = `<div class="figs">
        <figure>${img(r.prediction_png, 'Probability map')}<figcaption>Probability map (black 0, white 1)</figcaption></figure>
        <figure>${img(r.tirf_overlay_png, 'TIRF with edges')}<figcaption>TIRF with cell edges</figcaption></figure>
        <figure>${img(r.dic_overlay_png, 'DIC with edges')}<figcaption>DIC with cell edges</figcaption></figure></div>
        <button id="dlp" class="alt">Download probability map (PNG)</button>
        <p class="small mute">Load the downloaded map in the <a href="#/edges">edge lab</a> ("Load a prediction map") to try thresholds.</p>`;
      $('#dlp').addEventListener('click', async () => download('prediction.png', await (await fetch('data:image/png;base64,' + r.prediction_png)).blob()));
    } catch (err) { $('#msg').innerHTML = `<span class="err">${esc(err.message)}</span>`; }
    $('#go').disabled = !ready;
  });
});
