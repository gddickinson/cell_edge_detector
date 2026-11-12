// Architecture page: the U-Net of model.py (via unet_spec.py, checked against Keras) for a chosen input size.
import { route, app, api, esc, $, $$ } from './core.js';
import { unetSvg } from './unet_diagram.js';

const fmt = n => Number(n).toLocaleString('en-GB');

route('architecture', async (args, query) => {
  const info = await api('/info');
  const size = info.sizes.includes(+query.size) ? +query.size : info.default_input[0];
  const a = await api(`/architecture/${size}`);
  let lastBlock = '';
  const rows = a.layers.map(l => {
    const head = l.block !== lastBlock ? `<tr class="blk"><td colspan="4">${esc(l.block)}</td></tr>` : '';
    lastBlock = l.block;
    return `${head}<tr><td class="mono">${esc(l.name)}</td><td>${esc(l.kind)}</td>
      <td class="num">${l.shape.join(' × ')}</td><td class="num">${fmt(l.params)}</td></tr>`;
  }).join('');
  app().innerHTML = `<p class="kicker">model.py · UNetModel.build_unet</p><h1>U-Net architecture</h1>
    <p class="lead">A classic U-Net: four encoder levels of two 3×3 ReLU convolutions and 2×2 max pooling, a 1024-filter
      bridge, and four decoder levels that upsample with 2×2 transposed convolutions, concatenate the matching encoder
      output and convolve again. A 1×1 convolution with a sigmoid gives one probability per pixel.</p>
    <div class="seg" role="group" aria-label="Input size">${info.sizes.map(s =>
      `<button data-size="${s}" class="${s === size ? 'on' : ''}" aria-pressed="${s === size}">${s} px</button>`).join('')}</div>
    <div class="tiles">
      <div class="tile"><b>${fmt(a.total_params)}</b><span>parameters (the same for every input size)</span></div>
      <div class="tile"><b>${a.weights_mb_float32} MB</b><span>float32 weights</span></div>
      <div class="tile"><b>${a.input_shape.join('×')}</b><span>input (height × width × TIRF/DIC)</span></div>
      <div class="tile"><b>÷${a.divisible_by}</b><span>height and width must divide by ${a.divisible_by}</span></div></div>
    <div class="panel"><div class="scroll">${unetSvg(a)}</div>
      <div class="legend"><span><i style="background:var(--accent)"></i>encoder</span><span><i style="background:var(--accent2)"></i>bridge</span>
      <span><i style="background:var(--accent3)"></i>decoder</span><span>dropout ${a.dropout_rate} after encoder 4 and the bridge</span></div></div>
    <p class="small mute">Compiled with Adam (learning rate 1e-4) and Dice loss; the metric is the Dice coefficient. The
      default input is ${info.default_input.join('×')} (<code>UNetModel(input_shape=...)</code> changes it; the README
      example uses 768×768). Layer names are the Keras defaults. The table comes from <code>unet_spec.py</code>, which a
      test compares with the real Keras model (TensorFlow 2.20).</p>
    <h2>Layers</h2>
    <div class="scroll"><table><thead><tr><th>Layer</th><th>Type</th><th class="num">Output shape</th><th class="num">Parameters</th></tr></thead>
      <tbody>${rows}</tbody><tfoot><tr><td colspan="3"><b>Total</b></td><td class="num"><b>${fmt(a.total_params)}</b></td></tr></tfoot></table></div>`;
  $$('[data-size]').forEach(b => b.addEventListener('click', () => { location.hash = `#/architecture?size=${b.dataset.size}&keep=1`; }));
});
