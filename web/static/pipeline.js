// Training pipeline page: the steps of the Python workflow, with defaults read from the repo's code (/api/pipeline).
import { route, app, api, esc, list } from './core.js';

const pct = v => `${Math.round(v * 100)}%`;

route('pipeline', async () => {
  const p = await api('/pipeline');
  const d = p.dataset, f = d.fractions, [th, tw] = d.target_size;
  app().innerHTML = `<p class="kicker">data_loader · annotator · dataset · model · inference</p><h1>Training pipeline</h1>
    <p class="lead">How the Python workflow turns paired TIRF and DIC recordings into a trained edge detector. Numbers
      below are the defaults in the code; the driver is <code>EdgeDetectionApp</code> in <code>inference.py</code>.</p>
    <div class="steps">
      <div class="panel step"><h3>Find the image pairs</h3>
        <p><code>DataLoader.scan_directory</code> looks in <code>data/</code> for TIFF stacks named
          <code>${esc(p.patterns.tirf)}</code> (TIRF) and <code>${esc(p.patterns.dic)}</code> (DIC). Both lists are sorted
          and paired by position, so the n-th TIRF file goes with the n-th DIC file; frames are paired by index and a size
          mismatch is logged.</p></div>
      <div class="panel step"><h3>Annotate cells</h3>
        <p><code>AnnotationTool</code> shows TIRF, DIC and the mask side by side. Each lasso drawn on the DIC image fills
          the enclosed pixels into a binary mask; masks are saved per frame as
          <code>data/annotations/&lt;DIC name&gt;_frame&lt;n&gt;_mask.npy</code> (plus a PNG). The network learns these
          filled cell regions; edges are extracted from its output afterwards. The <a href="#/edges">edge lab</a> has the
          same lasso.</p></div>
      <div class="panel step"><h3>Build the dataset</h3>
        <p>Each annotated frame is resized to ${th}×${tw}, contrast-stretched to the ${p.stretch.percentile_low}st–${p.stretch.percentile_high}th
          percentile and stacked as two channels (TIRF, DIC). Masks are resized and re-binarised at 0.5. The samples are
          split with scikit-learn (random_state ${d.random_state}) and saved to <code>data/dataset/cell_edge_dataset.h5</code>.</p>
        <div class="split" role="img" aria-label="Split: ${pct(f.train)} train, ${pct(f.val)} validation, ${pct(f.test)} test">
          <span style="flex:${f.train};background:var(--accent)">train ${pct(f.train)}</span>
          <span style="flex:${f.val};background:var(--accent3)">val ${pct(f.val)}</span>
          <span style="flex:${f.test};background:var(--accent2)">test ${pct(f.test)}</span></div>
        <p class="small mute">test_split = ${d.test_split}, val_split = ${d.val_split} of the whole set (taken from the
          remainder as ${d.val_split}/${(1 - d.test_split).toFixed(1)}).</p></div>
      <div class="panel step"><h3>Augment</h3>
        <p>Training batches (not validation) pass through <code>get_data_generator</code>, which applies, per image:</p>
        <ul>${list(p.augmentation, a => `<li>${esc(a)}</li>`)}</ul></div>
      <div class="panel step"><h3>Train</h3>
        <p><code>train_model(batch_size=${p.train.batch_size}, epochs=${p.train.epochs})</code> builds the
          <a href="#/architecture">U-Net</a> and fits it; the run goes to <code>models/unet_training_&lt;timestamp&gt;/</code>
          and finishes with Dice and loss on the test split in <code>test_results.txt</code>.</p>
        <div class="scroll"><table><tbody>${list(p.training_settings, s =>
          `<tr><th>${esc(s.what)}</th><td>${esc(s.value)}</td></tr>`)}</tbody></table></div></div>
      <div class="panel step"><h3>Predict</h3>
        <p><code>EdgeDetector.detect_edges</code> preprocesses a frame pair the same way, runs the network and resizes
          the probability map back to the original frame size. <code>batch_process</code> does this for every frame and
          saves <code>_pred.npy</code>, <code>_edges.npy</code> and red-edge overlays (PNG).
          In this web app: the <a href="#/predict">Predict</a> page (local app, needs a trained model).</p></div>
      <div class="panel step"><h3>Extract edges</h3>
        <p><code>refine_edges(threshold=${p.refine.threshold}, edge_width=${p.refine.edge_width})</code> thresholds the
          map. With edge width 1 the edge is the binary mask minus its 3×3 erosion (the cell's inner boundary pixels);
          for wider edges it takes the iso-contour of the map at the threshold (marching squares) and paints a square of
          that width around each contour point. Try both in the <a href="#/edges">edge lab</a>.</p></div>
    </div>`;
});
