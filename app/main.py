"""FastAPI application object and router wiring.

Day 1 declared this module with no routes, by instruction. Day 3 mounts the two
routers the search flow needs. There is still no auth, no rate limiting and no
CORS configuration — see DECISIONS 18.8 for what that means before this is
exposed to anything but localhost.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import resumes, searches
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

app = FastAPI(
    title="Skillian",
    description="Job scraper and resume matcher.",
    version="0.3.0",
)

# Browser clients are served from a different origin in development (Vite on
# 5173, API on 8000), so without this every request fails at preflight.
# allow_credentials is False: the API has no auth and sets no cookies, and
# "*" methods/headers with credentials enabled is a combination browsers reject
# anyway. Origins come from config — see Settings.cors_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resumes.router)
app.include_router(searches.router)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness only.

    Deliberately does not touch Postgres or Ollama. A health check that fails
    when a dependency is down turns one outage into a restart loop; the run row
    and the 503s from the resume endpoints are where dependency failures should
    surface, with a message naming the fix.
    """
    return {"status": "ok", "version": app.version}
