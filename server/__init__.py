"""FastAPI layer exposing the Caption Forge backend to the React front-end.

This package is a thin, gradio-free adapter over the existing engines in
:mod:`src` (``loader``, ``captioner``, ``tagger``, ``quality``,
``embeddings``, ``deploy``, ``sqlite_store``...). It never imports the
Gradio UI modules (``src.ui_*``, ``src.events`` or the ``*_ui`` wrappers);
long-running batch loops are re-implemented as gradio-free job runners in
:mod:`server.runners`.

The app is assembled in :mod:`server.main`; run it with
``python -m uvicorn server.main:app``.
"""
