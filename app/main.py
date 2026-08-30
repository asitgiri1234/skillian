"""FastAPI application object.

Structure only on day 1: the app exists so deployment, settings wiring and the
import graph are established, but no routes are defined yet. Endpoints arrive
once there is something to serve.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Skillian",
    description="Job scraper and resume matcher.",
    version="0.1.0",
)
