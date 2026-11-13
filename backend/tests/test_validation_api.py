"""Validation HTTP and SQLite integration with Docker and GitHub simulated."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.api import validations as api
from app.config import Settings
from app.main import create_app
from app.validation_admission import ValidationAdmissionError
from app.validation_history import ValidationResultStore
from test_proposal_api import BASE, HEADERS, OTHER, login
from test_proposal_review_store import OWNER, decision
from test_validation_admission import admit, saved
from test_validation_runner import job


def app_for(saved, **changes):
    settings = Settings(_env_file=None, **{
        "enable_local_validation": True,
        "history_database_path": saved.path.parent / "history.db",
        "proposal_database_path": saved.path, **changes,
    })
    return create_app(settings, proposal_store=saved.store)


def client_for(app, host="127.0.0.1"):
    return TestClient(app, base_url=BASE, client=(host, 50000))


def post(client, saved, headers=HEADERS, **changes):
    return client.post("/api/validations", headers=headers, json={**saved.payload.model_dump(mode="json"), **changes})


def test_run_persists_comparison_and_owner_can_read_after_restart(job, saved):
    app = app_for(saved)
    proposal_id = saved.payload.proposal_id
    with client_for(app) as client:
        login(client, app)
        assert client.get("/api/validations", params={"proposal_id": proposal_id}).json()["results"] == []
        first = post(client, saved)
        assert first.status_code == 201, first.text
        second = post(client, saved)
        assert second.status_code == 201, second.text
        detail = second.json()
        assert detail["verdict"] == "verified" and detail["admission"]["state"] == "closed"
        assert detail["admission"]["binding"]["account_id"] == OWNER.id
        assert detail["baseline"]["failed"] == detail["patched"]["passed"] == 1
        assert detail["baseline"]["nodeids_sha256"] == detail["patched"]["nodeids_sha256"]
        url = "/api/validations/" + detail["admission"]["validation_id"]
        assert client.get(url).json() == detail
        listing = client.get("/api/validations", params={"proposal_id": proposal_id, "per_page": 1})
        assert listing.json() == {"results": [detail], "has_more": True, "next_page": 2}
        assert listing.headers["cache-control"] == second.headers["cache-control"] == "no-store"
        assert client.get("/api/validations", params={"proposal_id": proposal_id, "per_page": 1, "page": 2}).json()["results"] == [first.json()]
        schema = client.get("/openapi.json").json()
        assert any(p["name"] == "X-RepoFix-CSRF" for p in schema["paths"]["/api/validations"]["post"]["parameters"])
    restarted = create_app(app.state.settings)
    with client_for(restarted) as client:
        login(client, restarted, OTHER)
        assert client.get(url).status_code == 404
        assert client.get("/api/validations", params={"proposal_id": proposal_id}).status_code == 404
        login(client, restarted)
        assert client.get(url).json() == detail


@pytest.mark.parametrize(("changes", "host", "headers", "status"), [
    ({"enable_local_validation": False}, "127.0.0.1", HEADERS, 403),
    ({"environment": "production"}, "127.0.0.1", HEADERS, 403),
    ({}, "198.51.100.10", HEADERS, 403),
    ({}, "127.0.0.1", {}, 403),
    ({}, "127.0.0.1", {**HEADERS, "Origin": "https://other.example"}, 403),
    ({}, "127.0.0.1", [*HEADERS.items(), ("X-RepoFix-CSRF", "1")], 403),
])
def test_post_checks_local_opt_in_and_csrf_before_runner(job, saved, changes, host, headers, status):
    app = app_for(saved, **changes)
    with client_for(app, host) as client:
        login(client, app)
        assert post(client, saved, headers).status_code == status
    assert job.events == []


def test_no_session_or_browser_runtime_override_can_start_validation(job, saved):
    app = app_for(saved)
    with client_for(app) as client:
        assert post(client, saved).status_code == 401
        assert client.get("/api/validations", params={"proposal_id": saved.payload.proposal_id}).status_code == 401
        login(client, app)
        assert post(client, saved, command="pytest").status_code == 422
    assert job.events == []


def test_revoked_approval_and_process_busy_do_not_start_docker(job, saved):
    app = app_for(saved)
    with client_for(app) as client:
        login(client, app)
        with api._RUN_LOCK:
            assert post(client, saved).status_code == 429
        saved.store.record_decision(OWNER, decision(saved.source.proposal, decision="reject"), expected_revision=1)
        assert post(client, saved).status_code == 409
    assert job.events == []


def test_failed_insert_rolls_back_close_and_does_not_report_success(job, saved):
    with sqlite3.connect(saved.path) as connection:
        connection.execute("""
            CREATE TRIGGER fail_validation_result BEFORE INSERT ON fix_validation_results
            BEGIN SELECT RAISE(ABORT, 'test insert failure'); END
        """)
    app = app_for(saved)
    with client_for(app) as client:
        login(client, app)
        response = post(client, saved)
        assert response.status_code == 503 and "retained" in response.json()["detail"]
        assert "test insert failure" not in response.text
        assert not job.phases
        assert saved.admissions.get(OWNER.id, job.run_id).state == "running"
        assert client.get("/api/validations/" + job.run_id).status_code == 404


def test_corrupted_stored_verdict_is_not_returned(job, saved):
    result = job.run()
    with sqlite3.connect(saved.path) as connection:
        connection.execute(
            "UPDATE fix_validation_results SET detail_json=? WHERE validation_id=?",
            (result.model_copy(update={"verdict": "patched_tests_failed"}).model_dump_json(), job.run_id),
        )
    with pytest.raises(ValidationAdmissionError) as error:
        ValidationResultStore(saved.store).get(OWNER.id, job.run_id)
    assert error.value.code == "unavailable"


def test_version_three_upgrade_preserves_active_reservations_and_reviews(saved):
    reservation = admit(saved)
    review = saved.store.get_review(OWNER.id, saved.payload.proposal_id)
    saved.store.close()
    with sqlite3.connect(saved.path) as connection:
        connection.executescript("DROP TABLE fix_validation_results; PRAGMA user_version=3;")
    assert saved.admissions.get(OWNER.id, reservation.validation_id) == reservation
    assert saved.store.get_review(OWNER.id, saved.payload.proposal_id) == review
    assert ValidationResultStore(saved.store).list(OWNER.id, saved.payload.proposal_id).results == []
    with sqlite3.connect(saved.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
