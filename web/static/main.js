// Entry point: registers lazily loaded pages, sets the footer mode text and starts the router.
import { start, lazy, route, app, isStatic, $ } from './core.js';

const PAGES = { 'home.js': ['home'], 'architecture.js': ['architecture'], 'pipeline.js': ['pipeline'],
  'edges.js': ['edges'], 'predict.js': ['predict'], 'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: edge lab runs in your browser (JS port); predictions need the local app'
  : 'local app: Python server (predictions need TensorFlow and a trained model)';
start();
