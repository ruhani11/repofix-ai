"""Verify accepted repository links and rejected URL inputs."""

import pytest

from app.github_url import parse_github_repository_url


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/ruhani11/firmflow360",
        "https://github.com/ruhani11/firmflow360.git",
        "https://github.com/ruhani11/firmflow360/",
        "  https://github.com/ruhani11/firmflow360.git/  ",
    ],
)
def test_valid_repository_url_is_normalized(url):
    repository = parse_github_repository_url(url)

    assert repository.owner == "ruhani11"
    assert repository.name == "firmflow360"
    assert repository.canonical_url == "https://github.com/ruhani11/firmflow360"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://github.com/owner/repo",
        "https://example.com/owner/repo",
        "https://github.com.evil.example/owner/repo",
        "https://user:password@github.com/owner/repo",
        "https://github.com:443/owner/repo",
        "https://github.com/owner/repo/issues/1",
        "https://github.com/owner/repo?token=example",
        "https://github.com/owner/repo#readme",
        "https://github.com/owner/..",
        "https://github.com/owner/%2e%2e",
        "https://github.com/owner/re\npo",
    ],
)
def test_invalid_repository_url_is_rejected(url):
    with pytest.raises(ValueError):
        parse_github_repository_url(url)