"""HTTP layer. Routers only — no business logic lives here.

Every endpoint is a thin translation between JSON and the modules that do the
work (``app.matching.pipeline``, ``app.structure``). That boundary is what lets
the pipeline be tested without spinning up an ASGI app, and the CLI to keep
working without going through HTTP.
"""
