// SVG drawing of the U-Net from an /api/architecture summary: encoder down the left, bridge at the bottom,
// decoder up the right, dashed skip connections across. Box heights grow with the spatial size (log scale).
import { esc } from './core.js';

const lastConv = (layers, block) => layers.filter(l => l.block === block && l.kind === 'Conv2D').pop().shape;

export function unetSvg(arch) {
  const L = arch.layers, n = arch.encoder_filters.length;
  const enc = [...Array(n).keys()].map(i => lastConv(L, `encoder ${i + 1}`));
  const dec = [...Array(n).keys()].map(j => lastConv(L, `decoder ${j + 1}`));  // decoder 1 is the deepest
  const bridge = lastConv(L, 'bridge');
  const W = 760, H = 430, bw = 104, top = 44, step = 76;
  const hgt = s => 20 + 7 * Math.log2(s[0] / 16 + 1);
  const yOf = lvl => top + 34 + lvl * step;
  const xEnc = lvl => 16 + lvl * 70, xDec = lvl => W - 16 - bw - lvl * 70;
  const box = (x, y, s, cls, label) => {
    const hh = hgt(s);
    return `<rect class="${cls}" x="${x}" y="${y - hh / 2}" width="${bw}" height="${hh}" rx="4"/>
      <text x="${x + bw / 2}" y="${y - 2}" text-anchor="middle">${esc(label)}</text>
      <text class="sub" x="${x + bw / 2}" y="${y + 11}" text-anchor="middle">${s[0]}×${s[1]}×${s[2]}</text>`;
  };
  const arrow = (x1, y1, x2, y2) => `<path class="flow" d="M${x1} ${y1} L${x2} ${y2}"/>`;
  let g = '';
  enc.forEach((s, i) => {
    const y = yOf(i);
    g += `<path class="skip" d="M${xEnc(i) + bw} ${y} H${xDec(i)}"/>`;
    g += box(xEnc(i), y, s, 'enc', `encoder ${i + 1}`);
    if (i < n - 1) g += arrow(xEnc(i) + bw / 2, y + hgt(s) / 2, xEnc(i + 1) + bw / 2, yOf(i + 1) - hgt(enc[i + 1]) / 2 - 2);
  });
  const yb = yOf(n), xb = (W - bw) / 2;
  g += arrow(xEnc(n - 1) + bw / 2, yOf(n - 1) + hgt(enc[n - 1]) / 2, xb + 12, yb - hgt(bridge) / 2 - 2);
  g += box(xb, yb, bridge, 'mid', 'bridge');
  g += arrow(xb + bw - 12, yb - hgt(bridge) / 2, xDec(n - 1) + bw / 2, yOf(n - 1) + hgt(dec[0]) / 2 + 2);
  dec.forEach((s, j) => {
    const lvl = n - 1 - j, y = yOf(lvl);
    g += box(xDec(lvl), y, s, 'dec', `decoder ${j + 1}`);
    if (lvl > 0) g += arrow(xDec(lvl) + bw / 2, y - hgt(s) / 2, xDec(lvl - 1) + bw / 2, yOf(lvl - 1) + hgt(dec[j + 1]) / 2 + 2);
  });
  g += `<text x="16" y="${top - 16}">input ${arch.input_shape.join('×')} (TIRF, DIC)</text>`;
  g += `<text x="${W - 16}" y="${top - 16}" text-anchor="end">output ${arch.output_shape.join('×')} (1×1 conv, sigmoid)</text>`;
  g += `<text class="sub" x="${W / 2}" y="${top + 4}" text-anchor="middle">dashed: skip connection (concatenated)</text>`;
  g += `<text class="sub" x="16" y="${H - 8}">down: 2×2 max pool · each box: two 3×3 conv + ReLU</text>`;
  g += `<text class="sub" x="${W - 16}" y="${H - 8}" text-anchor="end">up: 2×2 transposed conv, stride 2</text>`;
  return `<svg class="unet" viewBox="0 0 ${W} ${H}" role="img" aria-label="U-Net diagram for a ${esc(arch.input_shape.join('×'))} input">
    <defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0 0 L8 4 L0 8 z" fill="#94a3c6"/></marker></defs>${g}</svg>`;
}
