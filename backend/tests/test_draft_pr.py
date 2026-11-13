"""Exercise explicit publication approval and GitHub writes without live requests."""

import json

import httpx
import pytest

from app.api import validations as api
from app.github_draft_pr import publish_draft_pr
from test_proposal_api import HEADERS, OTHER, login
from test_proposal_review_store import OWNER, decision
from test_validation_admission import saved
from test_validation_api import app_for, client_for
from test_validation_runner import job


TOKEN = "github_pat_fixture"
ROUTE = "/api/validations/draft-pr"
COMMIT = "7" * 40
TREE = "6" * 40


@pytest.fixture
def setup_pr(saved, job, monkeypatch):
    result = job.run()
    proposal = saved.source.proposal
    branch = f"repofix/{proposal.proposal_id}"
    prefix = f"/repos/{proposal.repository_full_name}"
    writes = []
    state = {"base": proposal.base_commit_sha, "branch": None, "pr": None, "wrong_tree": False, "actor": OWNER}
    old_tree = [item.model_dump() for item in saved.source.tree.entries]
    changes = {item.path: item for item in proposal.changes}
    new_tree = [entry for entry in old_tree if entry["path"] not in changes]
    new_tree.extend({"path": x.path, "mode": x.mode, "type": "blob", "sha": x.proposed_blob_sha}
                    for x in proposal.changes)
    for path in changes:
        parts = path.split("/")
        for length in range(1, len(parts)):
            directory = "/".join(parts[:length])
            if not any(item["path"] == directory for item in new_tree):
                new_tree.append({"path": directory, "mode": "040000", "type": "tree", "sha": TREE})

    def ref(name, sha):
        return {"ref": f"refs/heads/{name}", "object": {"type": "commit", "sha": sha}}

    def upstream(req):
        assert req.url.host == "api.github.com"
        assert req.headers["authorization"] == f"Bearer {TOKEN}"
        path = req.url.path
        if req.method == "GET":
            if path == "/user":
                return httpx.Response(200, json=state["actor"].model_dump())
            if path == prefix:
                return httpx.Response(200, json={
                    "id": proposal.repository_id, "full_name": proposal.repository_full_name,
                    "name": proposal.repository_full_name.split("/")[1], "owner": OWNER.model_dump(),
                    "private": False, "visibility": "public", "default_branch": "main",
                    "fork": False, "archived": False, "disabled": False,
                })
            if path == f"{prefix}/pulls":
                assert req.url.params["head"] == f"{OWNER.login}:{branch}"
                return httpx.Response(200, json=[state["pr"]] if state["pr"] else [])
            if path == f"{prefix}/git/ref/heads/main":
                return httpx.Response(200, json=ref("main", state["base"]))
            if path == f"{prefix}/git/ref/heads/{branch}":
                return httpx.Response(200, json=state["branch"]) if state["branch"] else httpx.Response(404)
            if path == f"{prefix}/git/trees/{proposal.tree_sha}":
                return httpx.Response(200, json={"sha": proposal.tree_sha, "tree": old_tree, "truncated": False})
            if path == f"{prefix}/git/trees/{TREE}":
                return httpx.Response(200, json={"sha": TREE, "tree": old_tree if state["wrong_tree"] else new_tree, "truncated": False})
            if path == f"{prefix}/git/commits/{COMMIT}":
                return httpx.Response(200, json={
                    "sha": COMMIT, "tree": {"sha": TREE}, "parents": [{"sha": proposal.base_commit_sha}],
                })
        else:
            assert req.method == "POST"
            payload = json.loads(req.content)
            writes.append(path.removeprefix(prefix))
            if path == f"{prefix}/git/trees":
                assert payload == {
                    "base_tree": proposal.tree_sha,
                    "tree": [{"path": c.path, "mode": c.mode, "type": "blob", "content": c.proposed_content}
                             for c in proposal.changes],
                }
                return httpx.Response(201, json={"sha": TREE})
            if path == f"{prefix}/git/commits":
                assert payload["tree"] == TREE and payload["parents"] == [proposal.base_commit_sha]
                return httpx.Response(201, json={
                    "sha": COMMIT, "tree": {"sha": TREE}, "parents": [{"sha": proposal.base_commit_sha}],
                })
            if path == f"{prefix}/git/refs":
                assert payload == {"ref": f"refs/heads/{branch}", "sha": COMMIT}
                state["branch"] = ref(branch, COMMIT)
                return httpx.Response(201, json=state["branch"])
            if path == f"{prefix}/pulls":
                assert payload["draft"] is True and payload["head"] == branch and payload["base"] == "main"
                assert "Original: 0 passed, 1 failed" in payload["body"]
                state["pr"] = {"number": 12, "draft": True, "state": "open",
                               "head": {"ref": branch, "sha": COMMIT, "repo": {"id": proposal.repository_id}},
                               "base": {"ref": "main", "sha": state["base"], "repo": {"id": proposal.repository_id}}}
                return httpx.Response(201, json=state["pr"])
        pytest.fail(f"Unexpected GitHub request: {req.method} {path}")

    transport = httpx.MockTransport(upstream)
    monkeypatch.setattr(api, "publish_draft_pr", lambda *args, **kwargs: publish_draft_pr(*args, **kwargs, transport=transport))
    payload = {**saved.payload.model_dump(), "validation_id": result.admission.validation_id, "confirm_publication": True}
    return app_for(saved, github_draft_pr_token=TOKEN), payload, writes, state


def test_publish_exact_patch_once_and_reopen_after_repeat(setup_pr):
    app, payload, writes, state = setup_pr
    with client_for(app) as client:
        login(client, app)
        response = client.post(ROUTE, headers=HEADERS, json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["url"] == f"https://github.com/demo-user/example/pull/12"
        assert response.json()["draft"] is True
        assert client.post(ROUTE, headers=HEADERS, json=payload).json() == response.json()
        assert writes == ["/git/trees", "/git/commits", "/git/refs", "/pulls"]
        assert TOKEN not in response.text


def test_retry_reuses_an_exact_branch_after_partial_publication(setup_pr):
    app, payload, writes, state = setup_pr
    with client_for(app) as client:
        login(client, app)
        assert client.post(ROUTE, headers=HEADERS, json=payload).status_code == 200
        state["pr"] = None
        writes.clear()
        assert client.post(ROUTE, headers=HEADERS, json=payload).status_code == 200
        assert writes == ["/pulls"]


def test_no_github_writes_without_consent_session_or_current_approval(setup_pr, saved):
    app, payload, writes, _ = setup_pr
    with client_for(app) as client:
        assert client.post(ROUTE, headers=HEADERS, json=payload).status_code == 401
        login(client, app, OTHER)
        assert client.post(ROUTE, headers=HEADERS, json=payload).status_code == 404
        login(client, app)
        assert client.post(ROUTE, headers={}, json=payload).status_code == 403
        assert client.post(ROUTE, headers=HEADERS, json={**payload, "confirm_publication": False}).status_code == 422
        assert client.post(ROUTE, headers=HEADERS, json={**payload, "validation_id": "2" * 32}).status_code == 404
        saved.store.record_decision(OWNER, decision(saved.source.proposal, decision="reject"), expected_revision=1)
        assert client.post(ROUTE, headers=HEADERS, json=payload).status_code == 409
        assert writes == []


def test_changed_default_branch_blocks_publication(setup_pr):
    app, payload, writes, state = setup_pr
    state["base"] = "8" * 40
    with client_for(app) as client:
        login(client, app)
        response = client.post(ROUTE, headers=HEADERS, json=payload)
        assert response.status_code == 409
        assert "default branch moved" in response.json()["detail"]
        assert writes == []


def test_unexpected_tree_never_creates_a_commit_or_branch(setup_pr):
    app, payload, writes, state = setup_pr
    state["wrong_tree"] = True
    with client_for(app) as client:
        login(client, app)
        assert client.post(ROUTE, headers=HEADERS, json=payload).status_code == 409
        assert writes == ["/git/trees"]


def test_foreign_token_cannot_publish(setup_pr):
    app, payload, writes, state = setup_pr
    state["actor"] = OTHER
    with client_for(app) as client:
        login(client, app)
        assert client.post(ROUTE, headers=HEADERS, json=payload).status_code == 403
        assert writes == []
