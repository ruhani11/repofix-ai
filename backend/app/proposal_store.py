"""Account-scoped SQLite proposals, separate from the investigation database."""

import json
from pathlib import Path
import re
import sqlite3
from threading import Lock
from typing import Callable

from pydantic import ValidationError

from app.github_oauth import GitHubIdentity
from app.proposal_decision import ProposalDecision, ProposalDecisionInput, bind_proposal_decision
from app.proposal_history import ProposalHistoryItem, ProposalHistoryPage, ProposalResponse
from app.proposal_review import (
    MAX_DECISION_RECORD_BYTES, ProposalDecisionEntry, ProposalDecisionPage,
    ProposalReviewError, ProposalReviewState,
)


MAX_PROPOSAL_RECORD_BYTES = 2 * 1024 * 1024
_IDENTIFIER = re.compile(r"[0-9a-f]{32}")


class ProposalStoreError(RuntimeError):
    def __init__(self):
        super().__init__("Fix proposal storage is unavailable.")


class ProposalStore:
    """Lazy app-owned proposals and human review log; never target execution."""

    def __init__(self, database_path: str | Path):
        if not isinstance(database_path, (str, Path)) or not str(database_path).strip():
            raise ValueError("A proposal database path is required.")
        self._path = str(database_path)
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None

    def _connect(self):
        if self._connection is not None:
            return self._connection
        connection = None
        try:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = sqlite3.connect(self._path, timeout=2.0, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, 1, 2, 3, 4}:
                raise ProposalStoreError()
            if version < 4:
                connection.executescript("""
                    BEGIN IMMEDIATE;
                    CREATE TABLE IF NOT EXISTS fix_proposals (
                        proposal_id TEXT PRIMARY KEY,
                        account_id INTEGER NOT NULL CHECK(account_id > 0),
                        investigation_run_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        repository_id INTEGER NOT NULL,
                        repository_full_name TEXT NOT NULL,
                        base_commit_sha TEXT NOT NULL,
                        title TEXT NOT NULL,
                        patch_sha256 TEXT NOT NULL,
                        total_duration_ms INTEGER NOT NULL,
                        detail_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS proposals_account_created
                        ON fix_proposals(account_id, created_at DESC, proposal_id DESC);
                    CREATE TABLE IF NOT EXISTS fix_proposal_decisions (
                        decision_id TEXT PRIMARY KEY,
                        proposal_id TEXT NOT NULL REFERENCES fix_proposals(proposal_id),
                        account_id INTEGER NOT NULL CHECK(account_id > 0),
                        revision INTEGER NOT NULL CHECK(revision > 0),
                        detail_json TEXT NOT NULL,
                        UNIQUE(proposal_id, revision)
                    );
                    CREATE TABLE IF NOT EXISTS fix_validation_admissions (
                        validation_id TEXT PRIMARY KEY,
                        account_id INTEGER NOT NULL CHECK(account_id > 0),
                        proposal_id TEXT NOT NULL REFERENCES fix_proposals(proposal_id),
                        state TEXT NOT NULL CHECK(state IN ('admitted', 'running', 'closed')),
                        detail_json TEXT NOT NULL
                    );
                    CREATE UNIQUE INDEX IF NOT EXISTS one_active_validation_per_account
                        ON fix_validation_admissions(account_id) WHERE state != 'closed';
                    CREATE TABLE IF NOT EXISTS fix_validation_results (
                        validation_id TEXT PRIMARY KEY REFERENCES fix_validation_admissions(validation_id),
                        account_id INTEGER NOT NULL CHECK(account_id > 0),
                        proposal_id TEXT NOT NULL REFERENCES fix_proposals(proposal_id),
                        completed_at TEXT NOT NULL,
                        detail_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS validation_results_account_proposal
                        ON fix_validation_results(account_id, proposal_id, completed_at DESC, validation_id DESC);
                    PRAGMA user_version=4;
                    COMMIT;
                """)
            self._connection = connection
            return connection
        except (OSError, sqlite3.Error, ProposalStoreError):
            if connection is not None:
                connection.close()
            raise ProposalStoreError() from None

    @staticmethod
    def _account(account_id):
        if type(account_id) is not int or not 0 < account_id <= 2**63 - 1:
            raise ProposalStoreError()

    def ensure_ready(self):
        """Test write access before a paid generation, without opening a long tx."""
        with self._lock:
            try:
                connection = self._connect()
                connection.execute("BEGIN IMMEDIATE")
                connection.rollback()
            except sqlite3.Error:
                raise ProposalStoreError() from None

    def _validation_transaction(self, callback):
        """Internal admission service bridge; callback must never run external I/O.

        Review lookup and durable job transition use this same database/write
        transaction. The callback can recheck the app's in-memory session after
        waiting, but must not call this store recursively or start Docker.
        """
        with self._lock:
            try:
                connection = self._connect()
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    result = callback(connection)
                return result
            except (sqlite3.Error, ValueError, TypeError, AttributeError):
                raise ProposalStoreError() from None

    def save(self, account_id: int, proposal: ProposalResponse) -> None:
        self._account(account_id)
        try:
            if not isinstance(proposal, ProposalResponse):
                raise ValueError()
            encoded = proposal.model_dump_json(warnings="error")
            if len(encoded.encode("utf-8")) > MAX_PROPOSAL_RECORD_BYTES:
                raise ValueError()
            # Do not trust shallow frozen models or nested mutable list contents.
            detail = ProposalResponse.model_validate_json(encoded)
            if not detail.proposal_saved:
                raise ValueError()
        except Exception:
            raise ProposalStoreError() from None
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO fix_proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (detail.proposal_id, account_id, detail.investigation_run_id,
                         detail.created_at.isoformat(), detail.repository_id,
                         detail.repository_full_name, detail.base_commit_sha,
                         detail.draft.title, detail.patch_sha256, detail.total_duration_ms, encoded),
                    )
            except sqlite3.Error:
                raise ProposalStoreError() from None

    def get(self, account_id: int, proposal_id: str) -> ProposalResponse | None:
        self._account(account_id)
        if not isinstance(proposal_id, str) or not _IDENTIFIER.fullmatch(proposal_id):
            raise ProposalStoreError()
        with self._lock:
            try:
                row = self._connect().execute(
                    "SELECT detail_json FROM fix_proposals WHERE account_id=? AND proposal_id=?",
                    (account_id, proposal_id),
                ).fetchone()
                if row is None:
                    return None
                return self._decode_proposal(row["detail_json"], proposal_id)
            except Exception:
                raise ProposalStoreError() from None

    @staticmethod
    def _decode_proposal(encoded, proposal_id):
        if not isinstance(encoded, str) or len(encoded.encode("utf-8")) > MAX_PROPOSAL_RECORD_BYTES:
            raise ValueError()
        detail = ProposalResponse.model_validate_json(encoded)
        if detail.proposal_id != proposal_id or not detail.proposal_saved:
            raise ValueError()
        return detail

    def _owned_proposal(self, connection, account_id, proposal_id):
        row = connection.execute(
            "SELECT detail_json FROM fix_proposals WHERE account_id=? AND proposal_id=?",
            (account_id, proposal_id),
        ).fetchone()
        if row is None:
            raise ProposalReviewError("not_found")
        return self._decode_proposal(row["detail_json"], proposal_id)

    @staticmethod
    def _review_key(account_id, proposal_id):
        if (
            type(account_id) is not int or not 0 < account_id <= 2**63 - 1
            or not isinstance(proposal_id, str) or not _IDENTIFIER.fullmatch(proposal_id)
        ):
            raise ProposalReviewError("invalid_input")

    @staticmethod
    def _decision_entry(row, account_id, proposal):
        if (
            type(row["revision"]) is not int or not 1 <= row["revision"] <= 2**63 - 1
            or len(row["detail_json"].encode("utf-8")) > MAX_DECISION_RECORD_BYTES
        ):
            raise ValueError()
        decision = ProposalDecision.model_validate_json(row["detail_json"])
        if (
            row["decision_id"] != decision.decision_id
            or row["account_id"] != account_id or decision.account_id != account_id
            or row["proposal_id"] != proposal.proposal_id or decision.proposal_id != proposal.proposal_id
            or decision.investigation_run_id != proposal.investigation_run_id
            or decision.repository_id != proposal.repository_id
            or decision.repository_full_name != proposal.repository_full_name
            or decision.base_commit_sha != proposal.base_commit_sha or decision.tree_sha != proposal.tree_sha
            or decision.patch_sha256 != proposal.patch_sha256 or decision.created_at < proposal.created_at
        ):
            raise ValueError()
        return ProposalDecisionEntry(revision=row["revision"], decision=decision)

    def _review_state(self, connection, account_id, proposal):
        row = connection.execute(
            "SELECT * FROM fix_proposal_decisions WHERE proposal_id=? ORDER BY revision DESC LIMIT 1",
            (proposal.proposal_id,),
        ).fetchone()
        entry = self._decision_entry(row, account_id, proposal) if row else None
        current = entry.decision if entry else None
        return ProposalReviewState(
            proposal_id=proposal.proposal_id, base_commit_sha=proposal.base_commit_sha,
            patch_sha256=proposal.patch_sha256, revision=entry.revision if entry else 0,
            status="unreviewed" if current is None else ("approved_for_validation" if current.decision == "approve_for_validation" else "rejected"),
            current_decision=current,
        )

    def get_review(self, account_id: int, proposal_id: str) -> ProposalReviewState:
        """Current state is a point-in-time view, not an execution capability."""
        self._review_key(account_id, proposal_id)
        with self._lock:
            try:
                connection = self._connect()
                proposal = self._owned_proposal(connection, account_id, proposal_id)
                return self._review_state(connection, account_id, proposal)
            except (sqlite3.Error, ProposalStoreError, ValueError, TypeError, AttributeError):
                raise ProposalReviewError("unavailable") from None

    def record_decision(
        self, identity: GitHubIdentity, payload: ProposalDecisionInput, *, expected_revision: int,
        authorize: Callable[[], None] | None = None,
    ) -> ProposalReviewState:
        """Owned lookup, revision compare and append share one write transaction.

        authorize is an internal session recheck after waiting for the database;
        it runs before the decision is inserted. It must not perform external I/O.
        """
        try:
            if not isinstance(identity, GitHubIdentity) or not isinstance(payload, ProposalDecisionInput):
                raise ValueError()
            identity = GitHubIdentity.model_validate(identity.model_dump(warnings="error"))
            payload = ProposalDecisionInput.model_validate(payload.model_dump(warnings="error"))
        except (ValidationError, ValueError, TypeError):
            raise ProposalReviewError("invalid_input") from None
        self._review_key(identity.id, payload.proposal_id)
        if type(expected_revision) is not int or not 0 <= expected_revision <= 2**63 - 2:
            raise ProposalReviewError("invalid_input")
        with self._lock:
            try:
                connection = self._connect()
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    proposal = self._owned_proposal(connection, identity.id, payload.proposal_id)
                    previous = self._review_state(connection, identity.id, proposal)
                    if previous.revision != expected_revision:
                        raise ProposalReviewError("revision_conflict")
                    if authorize is not None:
                        authorize()
                    decision = bind_proposal_decision(identity, proposal, payload)
                    encoded = decision.model_dump_json(warnings="error")
                    if len(encoded.encode("utf-8")) > MAX_DECISION_RECORD_BYTES:
                        raise ProposalReviewError("invalid_input")
                    connection.execute(
                        "INSERT INTO fix_proposal_decisions VALUES (?, ?, ?, ?, ?)",
                        (decision.decision_id, proposal.proposal_id, identity.id, previous.revision + 1, encoded),
                    )
                    state = self._review_state(connection, identity.id, proposal)
                # Commit has succeeded before the returned state claims a saved decision.
                return state
            except (sqlite3.Error, ProposalStoreError, ValueError, TypeError, AttributeError):
                raise ProposalReviewError("unavailable") from None

    def list_decisions(self, account_id: int, proposal_id: str, *, page: int = 1, per_page: int = 20) -> ProposalDecisionPage:
        self._review_key(account_id, proposal_id)
        if type(page) is not int or not 1 <= page <= 10000 or type(per_page) is not int or not 1 <= per_page <= 50:
            raise ProposalReviewError("invalid_input")
        with self._lock:
            try:
                connection = self._connect()
                proposal = self._owned_proposal(connection, account_id, proposal_id)
                rows = connection.execute(
                    "SELECT * FROM fix_proposal_decisions WHERE proposal_id=? ORDER BY revision DESC LIMIT ? OFFSET ?",
                    (proposal_id, per_page + 1, (page - 1) * per_page),
                ).fetchall()
                more = len(rows) > per_page
                return ProposalDecisionPage(
                    proposal_id=proposal_id, page=page, per_page=per_page,
                    decisions=[self._decision_entry(row, account_id, proposal) for row in rows[:per_page]],
                    has_more=more, next_page=page + 1 if more and page < 10000 else None,
                    limit_reached=more and page == 10000,
                )
            except (sqlite3.Error, ProposalStoreError, ValueError, TypeError, AttributeError):
                raise ProposalReviewError("unavailable") from None

    def list(self, account_id: int, *, page: int = 1, per_page: int = 20) -> ProposalHistoryPage:
        self._account(account_id)
        if type(page) is not int or not 1 <= page <= 10000 or type(per_page) is not int or not 1 <= per_page <= 50:
            raise ProposalStoreError()
        with self._lock:
            try:
                rows = self._connect().execute(
                    "SELECT proposal_id, investigation_run_id, created_at, repository_id, "
                    "repository_full_name, base_commit_sha, title, patch_sha256, total_duration_ms "
                    "FROM fix_proposals WHERE account_id=? "
                    "ORDER BY created_at DESC, proposal_id DESC LIMIT ? OFFSET ?",
                    (account_id, per_page + 1, (page - 1) * per_page),
                ).fetchall()
                items = [ProposalHistoryItem.model_validate_json(json.dumps(dict(row))) for row in rows[:per_page]]
                more = len(rows) > per_page
                return ProposalHistoryPage(
                    page=page, per_page=per_page, proposals=items, has_more=more,
                    next_page=page + 1 if more and page < 10000 else None,
                    limit_reached=more and page == 10000,
                )
            except Exception:
                raise ProposalStoreError() from None

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
