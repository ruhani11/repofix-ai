export class RepositoryApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.name = 'RepositoryApiError'
    this.status = status
  }
}

async function request(path, { signal, body } = {}) {
  let response
  try {
    response = await fetch(path, {
      credentials: 'same-origin',
      cache: 'no-store',
      headers: body
        ? { Accept: 'application/json', 'Content-Type': 'application/json' }
        : { Accept: 'application/json' },
      ...(body ? { method: 'POST', body: JSON.stringify(body) } : {}),
      signal,
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new RepositoryApiError('Could not connect to RepoFix AI. Check that the backend is running.')
  }

  const data = await response.json().catch((error) => {
    if (error?.name === 'AbortError') throw error
    return null
  })
  if (!response.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail :
      Array.isArray(data?.detail) ? data.detail.slice(0, 3).map((item) => item?.msg).filter(Boolean).join('; ') : ''
    throw new RepositoryApiError(detail || `Repository request failed (HTTP ${response.status}).`, response.status)
  }
  return data
}

export async function listInspectorRepositories({ signal } = {}) {
  const repositories = new Map()
  let accountLogin
  let privateAccess
  for (let page = 1; page <= 50; page++) {
    const data = await request(`/api/repositories/inspector?page=${page}`, { signal })
    if (!Array.isArray(data?.repositories) || typeof data.account_login !== 'string' ||
        typeof data.private_access !== 'boolean' ||
        (data.next_page !== null && data.next_page !== page + 1) ||
        (page > 1 && (data.account_login !== accountLogin || data.private_access !== privateAccess))) {
      throw new RepositoryApiError('The repository list changed or returned invalid data. Refresh the list.')
    }
    accountLogin = data.account_login
    privateAccess = data.private_access
    for (const repo of data.repositories) {
      if (!Number.isSafeInteger(repo?.id) || repo.id <= 0 || typeof repo.name !== 'string' ||
          typeof repo.full_name !== 'string' || !['public', 'private'].includes(repo.visibility)) {
        throw new RepositoryApiError('GitHub returned an invalid repository.')
      }
      repositories.set(repo.id, repo)
    }
    if (data.next_page === null || page === 50) {
      return { repositories: [...repositories.values()], accountLogin, privateAccess, limitReached: data.limit_reached || data.next_page !== null }
    }
  }
}

export async function inspectRepository(repository, selectedPaths = [], { signal } = {}) {
  const data = await request('/api/repositories/inspector/inspect', {
    signal,
    body: {
      repository_id: repository.id,
      repository_name: repository.name,
      selected_paths: selectedPaths,
    },
  })
  if (
    typeof data?.snapshot?.repository?.full_name !== 'string' ||
    typeof data?.snapshot?.commit_sha !== 'string' ||
    !Array.isArray(data?.tree?.entries) ||
    !Array.isArray(data?.files)
  ) {
    throw new RepositoryApiError('The API returned an unexpected inspection response.')
  }
  return data
}