"""Engine, session factory and the declarative Base."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all models (SQLAlchemy 2.x typed style)."""


def _build_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        # Ingestion runs are long and mostly idle while waiting on HTTP; a stale
        # pooled socket would surface as a random OperationalError mid-run.
        pool_pre_ping=True,
        # Cap connections: the CLI is single-threaded, FastAPI will add its own.
        pool_size=5,
        max_overflow=5,
        future=True,
    )


engine: Engine = _build_engine()

# expire_on_commit=False so ORM objects stay readable after commit — the CLI
# prints job rows after the transaction closes.
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception.

    Re-raises rather than swallowing — ingest.py owns the decision of what a
    failure means and is responsible for recording it on the ingestion_runs row.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
