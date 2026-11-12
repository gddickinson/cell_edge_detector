// Inputs for the edge lab: synthetic maps (labelled as such on the page) and images loaded by the user.

// Disc of value 1 on 0, like the synthetic prediction in tests/test_inference.py.
export function disc(h, w) {
  const out = new Float64Array(h * w), r = 0.3 * Math.min(h, w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    out[y * w + x] = (x - w / 2) ** 2 + (y - h / 2) ** 2 <= r * r ? 1 : 0;
  }
  return out;
}

// Two overlapping Gaussian blobs: a smooth map, so the threshold visibly moves the contour
// (the same shape as the 'blobs' case in scripts/make_edge_fixture.py, scaled to the grid).
export function blobs(h, w) {
  const out = new Float64Array(h * w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const X = x * 44 / w, Y = y * 36 / h;  // fixture grid is 44 x 36
    out[y * w + x] = Math.exp(-((X - 14.3) ** 2 + (Y - 12.1) ** 2) / 60)
      + 0.8 * Math.exp(-((X - 30.7) ** 2 + (Y - 22.6) ** 2) / 40);
  }
  return out;
}

// A fixed outline in grid coordinates, for adding a lasso without a pointer (keyboard users, demos).
export function exampleOutline(h, w) {
  const pts = [[4.3, 3.7], [30.6, 6.2], [38.2, 27.9], [20.4, 32.6], [9.1, 22.3], [15.7, 14.2]];  // fixture polygon
  return pts.map(([x, y]) => [x / 44 * (w - 1), y / 36 * (h - 1)]);
}

// Load an image file the browser can decode into greyscale values 0..255 on a grid at most maxSide pixels across.
export async function loadGrey(file, maxSide = 192) {
  const bmp = await createImageBitmap(file);
  const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const w = Math.max(16, Math.round(bmp.width * k)), h = Math.max(16, Math.round(bmp.height * k));
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  ctx.drawImage(bmp, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data, grey = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) grey[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  return { grey, h, w, original: [bmp.height, bmp.width], scaled: k < 1 };
}
