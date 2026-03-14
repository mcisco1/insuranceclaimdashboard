"""Database connection management for the insurance claims warehouse.

Provides a singleton SQLAlchemy engine and a session factory.  The default
database URL is read from ``config.DB_URL`` but can be overridden at
runtime (useful for testing or pointing at a different database).

Usage
-----
    from src.database.connection import get_engine, get_session

    engine = get_engine()                     # uses config.DB_URL
    Session = get_session()
    with Session() as session:
        ...
"""

from __future__ import annotations

import threading
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config import DB_URL

# ── Module-level singleton state ─────────────────────────────────────────

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None
_lock = threading.Lock()


# ── SQLite performance pragmas ───────────────────────────────────────────

def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """Enable WAL mode and foreign-key enforcement for SQLite connections."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()


# ── Public API ───────────────────────────────────────────────────────────

def get_engine(db_url: Optional[str] = None) -> Engine:
    """Return a singleton SQLAlchemy :class:`~sqlalchemy.engine.Engine`.

    Parameters
    ----------
    db_url : str, optional
        Database URL.  Defaults to ``config.DB_URL`` (an SQLite file inside
        ``data/warehouse/``).  If a new *db_url* is supplied that differs
        from the cached engine's URL, the old engine is disposed and a fresh
        one is created.

    Returns
    -------
    Engine
    """
    global _engine, _session_factory

    url = db_url or DB_URL

    with _lock:
        if _engine is not None and str(_engine.url) == url:
            return _engine

        # Dispose previous engine if URL changed
        if _engine is not None:
            _engine.dispose()
            _session_factory = None

        _engine = create_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            # SQLite does not support pool_size / max_overflow, but
            # create_engine silently ignores unsupported kwargs for
            # single-connection dialects.
        )

        # Attach SQLite pragmas (no-op for non-SQLite dialects)
        if url.startswith("sqlite"):
            event.listen(_engine, "connect", _set_sqlite_pragmas)

        return _engine


def get_session(db_url: Optional[str] = None) -> sessionmaker[Session]:
    """Return a reusable :class:`~sqlalchemy.orm.sessionmaker`.

    Parameters
    ----------
    db_url : str, optional
        Forwarded to :func:`get_engine` when the engine has not yet been
        created.

    Returns
    -------
    sessionmaker
        Call the returned factory (``Session()``) to open a new session.
    """
    global _session_factory

    engine = get_engine(db_url)

    with _lock:
        if _session_factory is None:
            _session_factory = sessionmaker(
                bind=engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )

    return _session_factory


def reset() -> None:
    """Dispose the cached engine and session factory.

    Primarily useful in test teardown.
    """
    global _engine, _session_factory

    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None
