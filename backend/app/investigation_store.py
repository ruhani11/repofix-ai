"""SQLite storage for completed, grounded investigations; no provider calls."""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from pydantic import ValidationError

from app.investigation_history import (
    InvestigationHistoryDetail, InvestigationHistoryItem, InvestigationHistoryPage,
    InvestigationRunResponse,
)
from app.investigation_input import InvestigationInput


MAX_HISTORY_RECORD_BYTES = 512 * 1024
_RUN_ID = re.compile(r"[0-9a-f]{32}")


class InvestigationStoreError(RuntimeError):
    """Storage failure with no SQLite message, file path or report payload."""


class InvestigationStore:
    """One lazy, lock-protected connection per app; a file survives restarts.

    :memory: is an explicit offline-test option. Construction does no disk I/O.
    Every query taking a run ID also takes a server-owned account ID.
    """

    def __init__(self, database_path: str | Path):
        if not isinstance(database_path, (str, Path)) or not str(database_path).strip():
            raise ValueError("A history database path is required.")
        self._path = str(database_path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        # Caller holds the lock, including during schema initialization.
        if self._connection is not None:
            return self._connection
        connection = None
        try:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = sqlite3.connect(
                self._path, timeout=2.0, check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1}:
                raise InvestigationStoreError("Investigation history is unavailable.")
            if version == 0:
                with connection:
                    connection.executescript("""
                        CREATE TABLE IF NOT EXISTS investigation_history (
                            run_id TEXT PRIMARY KEY,
                            account_id INTEGER NOT NULL CHECK(account_id > 0),
                            created_at TEXT NOT NULL,
                            repository_id INTEGER NOT NULL,
                            repository_name TEXT NOT NULL,
                            repository_full_name TEXT NOT NULL,
                            commit_sha TEXT NOT NULL,
                            report_status TEXT NOT NULL,
                            issue_summary TEXT NOT NULL,
                            total_duration_ms INTEGER NOT NULL,
                            detail_json TEXT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS history_account_created
                            ON investigation_history(account_id, created_at DESC, run_id DESC);
                        PRAGMA user_version=1;
                    """)
            self._connection = connection
            return connection
        except (OSError, sqlite3.Error, InvestigationStoreError):
            if connection is not None:
                connection.close()
            raise InvestigationStoreError("Investigation history is unavailable.") from None

    @staticmethod
    def _account(account_id: int) -> None:
        if type(account_id) is not int or not 0 < account_id <= 2**63 - 1:
            raise InvestigationStoreError("Invalid history account.")

    def ensure_ready(self) -> None:
        """Check current write access before admitting a paid investigation."""
        with self._lock:
            try:
                connection = self._connect()
                connection.execute("BEGIN IMMEDIATE")
                connection.rollback()
            except sqlite3.Error:
                raise InvestigationStoreError("Investigation history is unavailable.") from None

    def save(
        self, account_id: int, request: InvestigationInput,
        investigation: InvestigationRunResponse,
    ) -> None:
        self._account(account_id)
        if not isinstance(request, InvestigationInput) or not isinstance(investigation, InvestigationRunResponse):
            raise InvestigationStoreError("Invalid completed investigation.")
        if investigation.history_saved is not True:
            raise InvestigationStoreError("Invalid completed investigation.")
        detail = InvestigationHistoryDetail(
            created_at=datetime.now(timezone.utc), repository_id=request.repository_id,
            repository_name=request.repository_name,
            issue_description=request.issue_description, investigation=investigation,
        )
        encoded = detail.model_dump_json()
        if len(encoded.encode("utf-8")) > MAX_HISTORY_RECORD_BYTES:
            raise InvestigationStoreError("Investigation history record is too large.")
        with self._lock:
            try:
                connection = self._connect()
                with connection:
                    connection.execute(
                        "INSERT INTO investigation_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (investigation.run_id, account_id, detail.created_at.isoformat(),
                         detail.repository_id, detail.repository_name,
                         investigation.repository_full_name, investigation.commit_sha,
                         investigation.report.status, investigation.report.issue_summary,
                         investigation.total_duration_ms, encoded),
                    )
            except sqlite3.Error:
                raise InvestigationStoreError("Could not save investigation history.") from None

    def list(self, account_id: int, *, page: int = 1, per_page: int = 20) -> InvestigationHistoryPage:
        self._account(account_id)
        if type(page) is not int or not 1 <= page <= 10000 or type(per_page) is not int or not 1 <= per_page <= 50:
            raise InvestigationStoreError("Invalid history pagination.")
        with self._lock:
            try:
                rows = self._connect().execute(
                    "SELECT run_id, created_at, repository_id, repository_name, "
                    "repository_full_name, commit_sha, report_status, issue_summary, total_duration_ms "
                    "FROM investigation_history WHERE account_id = ? "
                    "ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?",
                    (account_id, per_page + 1, (page - 1) * per_page),
                ).fetchall()
                items = [
                    InvestigationHistoryItem.model_validate_json(json.dumps(dict(row)))
                    for row in rows[:per_page]
                ]
                has_more = len(rows) > per_page
                return InvestigationHistoryPage(
                    page=page, per_page=per_page, investigations=items,
                    has_more=has_more,
                    next_page=page + 1 if has_more and page < 10000 else None,
                    limit_reached=has_more and page == 10000,
                )
            except (sqlite3.Error, ValidationError, ValueError, TypeError):
                raise InvestigationStoreError("Could not read investigation history.") from None

    def get(self, account_id: int, run_id: str) -> InvestigationHistoryDetail | None:
        self._account(account_id)
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise InvestigationStoreError("Invalid history identifier.")
        with self._lock:
            try:
                row = self._connect().execute(
                    "SELECT detail_json FROM investigation_history WHERE account_id = ? AND run_id = ?",
                    (account_id, run_id),
                ).fetchone()
                if row is None:
                    return None
                if len(row["detail_json"].encode("utf-8")) > MAX_HISTORY_RECORD_BYTES:
                    raise InvestigationStoreError("Invalid stored investigation.")
                detail = InvestigationHistoryDetail.model_validate_json(row["detail_json"])
                if detail.investigation.run_id != run_id or not detail.investigation.history_saved:
                    raise InvestigationStoreError("Invalid stored investigation.")
                return detail
            except (sqlite3.Error, ValidationError, ValueError, TypeError, AttributeError):
                raise InvestigationStoreError("Could not read investigation history.") from None

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
