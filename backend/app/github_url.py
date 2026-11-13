"""Validate repository links without contacting GitHub."""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class GitHubRepository:
    owner: str
    name: str

    @property
    def canonical_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


def parse_github_repository_url(url: str) -> GitHubRepository:
    """Accept a GitHub HTTPS repository URL and return its owner/name."""
    cleaned_url = url.strip()

    if not cleaned_url or len(cleaned_url) > 300:
        raise ValueError("Repository URL must contain 1 to 300 characters.")

    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in cleaned_url
    ):
        raise ValueError("Repository URL contains whitespace or control characters.")

    parsed_url = urlsplit(cleaned_url)

    if parsed_url.scheme != "https" or parsed_url.netloc.lower() != "github.com":
        raise ValueError("Use an HTTPS repository URL from github.com.")

    if parsed_url.query or parsed_url.fragment:
        raise ValueError("Repository URL must not contain a query or fragment.")

    match = re.fullmatch(
        r"/([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9_.-]{1,104})/?",
        parsed_url.path,
    )
    if match is None:
        raise ValueError("Expected https://github.com/OWNER/REPOSITORY.")

    owner, name = match.groups()
    name = name.removesuffix(".git")

    if not name or name in {".", ".."} or len(name) > 100:
        raise ValueError("Invalid repository name.")

    return GitHubRepository(owner=owner, name=name)