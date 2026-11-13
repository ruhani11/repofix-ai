"""Application entry point and composition."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.investigations import router as investigations_router
from app.api.repositories import router as repositories_router
from app.api.proposals import router as proposals_router
from app.api.validations import router as validations_router
from app.auth_store import AuthStore
from app.config import Settings, get_settings
from app.investigation_store import InvestigationStore
from app.proposal_store import ProposalStore


def create_app(
    settings: Settings | None = None, *, history_store: InvestigationStore | None = None,
    proposal_store: ProposalStore | None = None,
) -> FastAPI:
    """Allow explicit configuration in tests without external services."""
    configured_settings = settings if settings is not None else get_settings()
    store = history_store if history_store is not None else InvestigationStore(
        configured_settings.history_database_path,
    )
    if not isinstance(store, InvestigationStore):
        raise ValueError("An InvestigationStore instance is required.")
    proposals = proposal_store if proposal_store is not None else ProposalStore(
        configured_settings.proposal_database_path,
    )
    if not isinstance(proposals, ProposalStore):
        raise ValueError("A ProposalStore instance is required.")

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        try:
            yield
        finally:
            try:
                await asyncio.to_thread(store.close)
            finally:
                await asyncio.to_thread(proposals.close)

    application = FastAPI(
        title=configured_settings.app_name,
        version=__version__,
        description=(
            "GitHub account connection, read-only repository inspection and "
            "grounded AI investigations and saved fix proposals for RepoFix AI."
        ),
        lifespan=lifespan,
    )
    application.state.settings = configured_settings
    # One shared store per app process: login and callback use the same records.
    application.state.auth_store = AuthStore()
    application.state.investigation_store = store
    application.state.proposal_store = proposals
    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(repositories_router)
    application.include_router(investigations_router)
    application.include_router(proposals_router)
    application.include_router(validations_router)
    return application


app = create_app()
