"""Web interface for Cell Edge Detector (FastAPI + plain ES-module front end).

Contents:
    app.py      FastAPI app: GET /api/info, /api/pipeline, /api/architecture/{size}, /api/models;
                POST /api/preview, /api/predict, /api/refine; STATIC_ENDPOINTS for the static site
    predict.py  upload reading, trained-model loading and PNG encoding around DataLoader / EdgeDetector
    static/     front end (index.html, core.js router, one module per page, edge_engine.js JS port)
"""
