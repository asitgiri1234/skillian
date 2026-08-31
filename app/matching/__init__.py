"""Resume-to-job matching: chunking, scoring, explanation, orchestration.

The package boundary that matters is between :mod:`app.matching.scorer` and
everything else. The scorer is pure arithmetic over plain dataclasses — no ORM,
no network, no LLM — which is what lets it be tested exhaustively and run over a
few hundred jobs in under a second. Anything that needs a database session or a
model call lives in :mod:`app.matching.pipeline`.
"""
