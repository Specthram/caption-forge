"""Gradio-free job bodies for the long-running backend operations.

Each function returns a ``run(progress)`` callable suitable for
:meth:`server.jobs.JobManager.submit`. They call the pure ``src`` engines
directly and report progress through the injected :class:`~server.jobs.
Progress` reporter — never importing the Gradio ``*_ui`` wrappers.
"""
