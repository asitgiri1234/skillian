"""Alembic environment.

Pulls the database URL from app.config rather than alembic.ini so migrations and
the application can never disagree about which database they are pointed at.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db import Base

# Importing the models module is what registers every table on Base.metadata;
# without it autogenerate would see an empty schema and propose dropping
# everything.
import app.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def render_item(type_: str, obj: object, autogen_context: object) -> str | bool:
    """Render pgvector columns correctly during autogenerate.

    Without this, autogenerate emits a bare ``pgvector.sqlalchemy.vector.VECTOR``
    reference and no matching import, producing a migration that dies with
    NameError the first time it runs. Returning False defers to Alembic's default
    rendering for every other type.
    """
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({obj.dim})"
    return False


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (alembic upgrade --sql)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Detect column type changes on autogenerate, not just added/dropped ones.
        compare_type=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # NullPool: a migration process is short-lived and single-connection,
        # so a pool would only hold sockets open past the work.
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_item=render_item,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
