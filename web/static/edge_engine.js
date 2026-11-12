// JS port of the repo's pure image steps (no DOM), checked against Python by edge_engine.test.mjs:
//   stretch()      DataLoader.preprocess_for_visualization: clip to the 1st..99th percentile, scale to 0..1
//   polygonMask()  annotator.polygon_mask: pixels whose centre lies inside the lasso polygon
//   refineEdges()  EdgeDetector.refine_edges: threshold, then a 1-px inner boundary (binary - erosion) for
//                  edgeWidth 1, or marching-squares contour points (skimage find_contours), rounded and
//                  widened to edgeWidth pixels, for edgeWidth > 1.
// Images are row-major flat arrays of length h * w.

// np.percentile with the default 'linear' method.
export function percentile(values, q) {
  const s = Float64Array.from(values).sort();
  const idx = (q / 100) * (s.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

export function stretch(img, pLow = 1, pHigh = 99) {
  const low = percentile(img, pLow), high = percentile(img, pHigh);
  const out = new Float32Array(img.length);
  if (!(high - low > 0)) return out;
  for (let i = 0; i < img.length; i++) out[i] = (Math.min(Math.max(img[i], low), high) - low) / (high - low);
  return out;
}

// Even-odd ray casting test of pixel centres (x = column, y = row), as matplotlib Path.contains_points.
export function polygonMask(verts, h, w) {
  const out = new Uint8Array(h * w);
  const n = verts.length;
  if (n < 3) return out;
  for (let y = 0; y < h; y++) {
    const xs = [];  // x where the polygon edges cross this row
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const [xi, yi] = verts[i], [xj, yj] = verts[j];
      if ((yi > y) !== (yj > y)) xs.push(xi + (y - yi) * (xj - xi) / (yj - yi));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const x0 = Math.max(0, Math.ceil(xs[k])), x1 = Math.min(w - 1, Math.ceil(xs[k + 1]) - 1);
      for (let x = x0; x <= x1; x++) if (x > xs[k] && x < xs[k + 1]) out[y * w + x] = 1;
    }
  }
  return out;
}

// numpy rounding: halves go to the nearest even integer.
export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// Every point where the level crosses a grid edge (the vertices skimage.measure.find_contours joins into lines).
export function contourPoints(pred, h, w, level) {
  const pts = [];
  const frac = (a, b) => (a === b ? 0 : (level - a) / (b - a));
  for (let r = 0; r < h - 1; r++) {
    for (let c = 0; c < w - 1; c++) {
      const ul = pred[r * w + c], ur = pred[r * w + c + 1], ll = pred[(r + 1) * w + c], lr = pred[(r + 1) * w + c + 1];
      const a = ul > level, b = ur > level, d = ll > level, e = lr > level;
      if (a === b && b === d && d === e) continue;
      if (a !== b) pts.push([r, c + frac(ul, ur)]);
      if (d !== e) pts.push([r + 1, c + frac(ll, lr)]);
      if (a !== d) pts.push([r + frac(ul, ll), c]);
      if (b !== e) pts.push([r + frac(ur, lr), c + 1]);
    }
  }
  return pts;
}

export function refineEdges(pred, h, w, threshold = 0.5, edgeWidth = 1) {
  const out = new Uint8Array(h * w);
  if (edgeWidth === 1) {
    // binary - erosion(binary, 3x3); skimage pads with mode='reflect', i.e. the border pixel repeats itself
    const bin = Uint8Array.from(pred, v => (v > threshold ? 1 : 0));
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        if (!bin[r * w + c]) continue;
        let m = 1;
        for (let dr = -1; dr <= 1 && m; dr++) {
          const rr = Math.min(h - 1, Math.max(0, r + dr));
          for (let dc = -1; dc <= 1; dc++) {
            const cc = Math.min(w - 1, Math.max(0, c + dc));
            if (!bin[rr * w + cc]) { m = 0; break; }
          }
        }
        out[r * w + c] = 1 - m;
      }
    }
    return out;
  }
  const lo = -Math.floor((edgeWidth - 1) / 2), hi = Math.floor(edgeWidth / 2);
  for (const [pr, pc] of contourPoints(pred, h, w, threshold)) {
    const x = roundHalfEven(pr), y = roundHalfEven(pc);
    if (x < 0 || x >= h || y < 0 || y >= w) continue;
    for (let dx = lo; dx <= hi; dx++) {
      for (let dy = lo; dy <= hi; dy++) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < h && ny >= 0 && ny < w) out[nx * w + ny] = 1;
      }
    }
  }
  return out;
}

// Summary numbers for an edge map and the mask it came from.
export function edgeStats(pred, edges, threshold) {
  let inside = 0, edge = 0;
  for (let i = 0; i < pred.length; i++) { inside += pred[i] > threshold; edge += edges[i]; }
  return { inside, edge };
}
