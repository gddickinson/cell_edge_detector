// About page: what the project is, how to run the Python workflow and this web app, credits (from the README).
import { route, app, api, esc } from './core.js';

route('about', async () => {
  const info = await api('/info');
  app().innerHTML = `<p class="kicker">About</p><h1>Cell Edge Detection System</h1>
    <p class="lead">A U-Net that detects cell edges in TIRF (total internal reflection fluorescence) and DIC (differential
      interference contrast) microscopy images. You load and view TIRF and DIC image stacks, annotate cell boundaries to
      make training data, train the network, apply it to new images, and export the detected edges.</p>
    <h2>Run the Python workflow</h2>
    <pre>pip install -r requirements.txt     # TensorFlow, OpenCV, scikit-image, h5py, ...
python inference.py                 # scans data/ and opens the annotation tool</pre>
    <pre>from inference import EdgeDetectionApp
app = EdgeDetectionApp()            # data in ./data: *piezo*.tif (TIRF), *DIC*.tif (DIC)
app.initialize()
app.annotate_images()               # lasso annotation, masks saved to data/annotations/
history, out_dir = app.train_model(epochs=30)
app.load_trained_model('models/unet_training_YYYYMMDD-HHMMSS/best_model.h5')
prediction, edges, tirf_overlay, dic_overlay = app.process_image_pair(index=0, frame=0)
app.batch_process()                 # .npy maps and PNG overlays for every frame</pre>
    <h2>Run the web app</h2>
    <pre>pip install -r requirements-web.txt   # add tensorflow for the Predict page
python run_web.py                     # http://127.0.0.1:8147</pre>
    <p>The web app reads the repository's own code: the architecture table is produced by <code>unet_spec.py</code>
      (checked against the Keras model by a test), the pipeline defaults are read from the function signatures, and the
      Predict page calls <code>EdgeDetector</code> with a model from <code>models/</code>. On this static site the edge
      lab runs a JavaScript port of <code>refine_edges</code>, <code>polygon_mask</code> and the percentile stretch
      (<code>edge_engine.js</code>), which a node test checks against the Python output; in the local app the lab also
      runs the Python function and reports whether the two agree.</p>
    <p>The repository contains no microscopy data and no trained weights (about ${info.weights_mb_float32} MB once
      trained), so the site shows no predictions; the test maps in the edge lab are synthetic and labelled as such.
      Edge widths and positions are in pixels; the code does not use a physical pixel size.</p>
    <h2>Credits</h2>
    <p>Built with AI assistance from <a href="https://claude.com/" target="_blank" rel="noopener">Claude (Anthropic)</a>.
      Licence: MIT. Source: <a href="https://github.com/gddickinson/${esc(info.repo)}" target="_blank" rel="noopener">github.com/gddickinson/${esc(info.repo)}</a>.</p>`;
});
