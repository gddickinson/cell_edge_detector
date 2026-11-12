// Node test: edge_engine.js must reproduce the repo's Python on tests/fixtures/edge_fixture.json
// (written by scripts/make_edge_fixture.py). Run: node web/static/edge_engine.test.mjs
import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
import { stretch, polygonMask, refineEdges, roundHalfEven } from './edge_engine.js';

const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/edge_fixture.json', import.meta.url)));
const { height: h, width: w } = fx;
const on = arr => [...arr.keys()].filter(i => arr[i]);
let n = 0;

// percentile stretch (Python works in float32: allow 1e-6)
const s = stretch(fx.raw);
fx.stretch.forEach((v, i) => assert.ok(Math.abs(s[i] - v) < 1e-6, `stretch[${i}] ${s[i]} vs ${v}`));
n++;

assert.deepEqual(on(polygonMask(fx.polygon, h, w)), fx.polygon_mask, 'polygon mask');
n++;

for (const [key, want] of Object.entries(fx.edges)) {
  const [name, thr, width] = key.split('|');
  const got = on(refineEdges(fx.maps[name], h, w, +thr, +width));
  assert.deepEqual(got, want, `refineEdges ${key}: ${got.length} vs ${want.length} pixels`);
  n++;
}

assert.deepEqual([0.5, 1.5, 2.5, 9.5, 2.4, 2.6].map(roundHalfEven), [0, 2, 2, 10, 2, 3]);
n++;
console.log(`edge_engine: ${n} checks passed`);
