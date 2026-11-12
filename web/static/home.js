// Home page: hero, headline numbers computed from the repo's code, and links to the pages.
import { route, app, api, esc, isStatic } from './core.js';

const fmt = n => Number(n).toLocaleString('en-GB');

route('home', async () => {
  const info = await api('/info');
  const [h, w, c] = info.default_input;
  app().innerHTML = `<section class="hero"><p class="kicker">TIRF · DIC · U-Net segmentation</p>
    <h1 class="glow">Cell Edge Detector</h1>
    <p class="lead">${esc(info.description)}</p>
    <a class="btn" href="#/edges">Open the edge lab</a><a class="btn alt" href="#/architecture">See the network</a></section>
    <div class="tiles">
      <div class="tile"><b>${fmt(info.total_params)}</b><span>trainable parameters</span></div>
      <div class="tile"><b>${info.weights_mb_float32} MB</b><span>of float32 weights once trained</span></div>
      <div class="tile"><b>${h}×${w}×${c}</b><span>default input: TIRF + DIC channels</span></div>
      <div class="tile"><b>${info.conv_layers}</b><span>convolution layers over ${info.levels} resolution levels</span></div></div>
    <div class="callout small">No trained weights are committed to the repository (a trained model would be about
      ${info.weights_mb_float32} MB), so this ${isStatic() ? 'site' : 'app'} shows no model predictions of its own. The
      architecture and pipeline pages are generated from the code; the edge lab runs the repo's post-processing on masks
      you draw or images you load; the Predict page runs a model you have trained, in the local app.</div>
    <div class="grid">
      <a class="card" href="#/architecture"><b>Architecture</b><span>The U-Net layer by layer: encoder, bridge, decoder,
        skip connections, output shapes and parameter counts for 256, 512 or 768 px inputs.</span></a>
      <a class="card" href="#/pipeline"><b>Training pipeline</b><span>From TIRF/DIC TIFF stacks to lasso annotations,
        the HDF5 dataset, augmentation, Dice-loss training and inference.</span></a>
      <a class="card" href="#/edges"><b>Edge lab</b><span>Draw a cell outline (as in the annotation tool) or load a
        prediction map, then threshold it and extract edges exactly as <code>refine_edges</code> does.</span></a>
      <a class="card" href="#/predict"><b>Predict</b><span>Local app: upload a TIRF and a DIC stack, pick a trained
        model from models/, and get the probability map and edge overlays.</span></a>
      <a class="card" href="#/about"><b>About</b><span>What the project is, how to run the Python workflow and this web
        app, and credits.</span></a>
    </div>`;
});
