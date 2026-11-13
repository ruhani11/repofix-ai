# RepoFix AI

RepoFix AI investigates bugs in GitHub repositories, proposes a patch with regression tests, and checks whether those tests fail on the original code and pass after the fix. A developer reviews the diff and approves validation before the tests run, then separately confirms publication of a draft pull request.

## Live demo

[View the draft PR created by RepoFix AI](https://github.com/ruhani11/repofix-demo/pull/1).

The demo API rejected orders with a quantity of 1. RepoFix identified the boundary check in `orders_api.py`, proposed a fix, and added regression tests.

| Version | Passed | Failed |
| --- | ---: | ---: |
| Original | 2 | 1 |
| Patched | 3 | 0 |

RepoFix ran the same three tests in its local Docker sandbox. The draft PR contains the verified change and its test results; the demo's default branch was left untouched.

## Workflow

1. **Sign in with GitHub.** The app opens GitHub Investigation. The same session also works in Validation.
2. **Investigate.** Select a repository and describe the issue. RepoFix reports a likely cause with source citations or explains why it lacks evidence.
3. **Generate and review.** Open the proposed fix in Validation, inspect the diff and new tests, and approve the exact patch for testing.
4. **Validate.** Run the same selected tests against the original and patched code in separate Docker containers. Reopen saved comparisons from Validation.
5. **Create a draft PR.** Confirm publication of the verified patch. RepoFix creates a separate GitHub branch and opens a draft PR for review.

## Local setup

Requirements: Python 3.11 or newer, Node.js 20.19+ on the 20.x line or 22.12+, a GitHub OAuth app, and an API key for the configured AI provider. Docker Desktop with Linux containers is required for new validation runs.

### 1. Configure and start the backend

From the project root in Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Use `http://127.0.0.1:5173/` as the GitHub OAuth app's homepage and `http://127.0.0.1:8000/api/auth/github/callback` as its callback. In `backend/.env`, set `REPOFIX_GITHUB_CLIENT_ID` and `REPOFIX_GITHUB_CLIENT_SECRET`. Set `REPOFIX_AI_PROVIDER`, `REPOFIX_AI_MODEL`, and the corresponding `REPOFIX_OPENAI_API_KEY` or `REPOFIX_GEMINI_API_KEY` for a model your account can use.

For optional private repository inspection, set `REPOFIX_GITHUB_INSPECTOR_TOKEN` to a fine-grained GitHub token with **Contents: Read-only** for the selected repositories. For draft PRs, use a separate `REPOFIX_GITHUB_DRAFT_PR_TOKEN` scoped to the target repository, with **Contents: Read and write** and **Pull requests: Read and write**. Keep real credentials in `backend/.env`.

To enable new validations, start Docker Desktop and run these commands from `backend`:

```powershell
docker build --tag repofix-python-validation:v1 --file .\sandbox\python\Dockerfile .\sandbox\python
.\.venv\Scripts\python.exe -m app.sandbox_runtime_check
$env:REPOFIX_ENABLE_LOCAL_VALIDATION = "true"
```

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

### 2. Start the frontend

In a second terminal, from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open [RepoFix AI](http://127.0.0.1:5173/) and choose **Continue with GitHub**. Keep both services on `127.0.0.1` for local sign-in. The API docs are at [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Checks and scope

Run `python -m pytest` in the activated backend environment and `npm run build` from `frontend`.

The full fix-to-PR workflow currently targets public repositories owned by the signed-in account. Private repositories support inspection with a configured read token. The bundled validator supports Python/pytest in a restricted local Docker container, and a verified result describes the selected tests rather than the entire repository. Local validation and draft PR creation are available in development mode.
