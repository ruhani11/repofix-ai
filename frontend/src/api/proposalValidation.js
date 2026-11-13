export class WorkbenchApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.status = status
  }
}

async function request(path, { body, signal, interruptedMessage } = {}) {
  const write = body !== undefined
  let response
  try {
    response = await fetch(path, {
      method: write ? 'POST' : 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: {
        Accept: 'application/json',
        ...(write ? { 'Content-Type': 'application/json', 'X-RepoFix-CSRF': '1' } : {}),
      },
      body: write ? JSON.stringify(body) : undefined,
      signal,
    })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new WorkbenchApiError(interruptedMessage ?? (write
      ? 'The request was interrupted. Reload the review and saved results before retrying.'
      : 'Could not reach RepoFix AI. Check that the backend is running.'))
  }
  const data = await response.json().catch((error) => {
    if (error?.name === 'AbortError') throw error
    return null
  })
  if (!response.ok) {
    throw new WorkbenchApiError(
      typeof data?.detail === 'string' ? data.detail : `Request failed (HTTP ${response.status}).`,
      response.status,
    )
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new WorkbenchApiError('The backend returned an unexpected response.', response.status)
  }
  return data
}

const proposalPath = (id) => `/api/proposals/${encodeURIComponent(id)}`

export const currentAccount = (signal) => request('/api/auth/me', { signal })
export const listProposals = (page, signal) => request(`/api/proposals?page=${page}`, { signal })
export const getProposal = (id, signal) => request(proposalPath(id), { signal })
export const getReview = (id, signal) => request(`${proposalPath(id)}/decision`, { signal })
export const listValidations = (id, page = 1, signal) => request(
  `/api/validations?${new URLSearchParams({ proposal_id: id, page })}`, { signal },
)

export const runInvestigation = (repository, issueDescription) => request('/api/investigations/run', {
  body: {
    repository_id: repository.id,
    repository_name: repository.name,
    issue_description: issueDescription,
  },
  interruptedMessage: 'The investigation response was interrupted. It may have been saved; check before starting another run.',
})

export const generateFixProposal = (runId) => request('/api/proposals/generate', {
  body: { investigation_run_id: runId },
  interruptedMessage: 'The proposal response was interrupted. Refresh saved proposals in Validation before generating again.',
})

export function reviewMatches(proposal, review) {
  return Boolean(proposal && review
    && proposal.proposal_id === review.proposal_id
    && proposal.base_commit_sha === review.base_commit_sha
    && proposal.patch_sha256 === review.patch_sha256
    && Number.isSafeInteger(review.revision) && review.revision >= 0)
}

function reviewBinding(proposal, review) {
  if (!reviewMatches(proposal, review)) {
    throw new WorkbenchApiError('The review no longer matches this patch. Reload the proposal.', 409)
  }
  return {
    proposal_id: proposal.proposal_id,
    base_commit_sha: proposal.base_commit_sha,
    patch_sha256: proposal.patch_sha256,
    expected_revision: review.revision,
  }
}

export async function recordDecision(proposal, review, decision) {
  return request('/api/proposals/decision', { body: { ...reviewBinding(proposal, review), decision } })
}

export async function startValidation(proposal, review) {
  const binding = reviewBinding(proposal, review)
  const id = review.current_decision?.decision_id
  if (review.status !== 'approved_for_validation' || review.revision < 1 || !/^[0-9a-f]{32}$/.test(id ?? '')) {
    throw new WorkbenchApiError('Approve this proposal before running validation.', 409)
  }
  return request('/api/validations', { body: { ...binding, decision_id: id } })
}

export async function createDraftPullRequest(proposal, review, validation) {
  const binding = reviewBinding(proposal, review)
  const data = await request('/api/validations/draft-pr', {
    body: {
      ...binding,
      decision_id: review.current_decision?.decision_id,
      validation_id: validation.admission.validation_id,
      confirm_publication: true,
    },
    interruptedMessage: 'GitHub may have received the request. Check the proposal branch and pull requests before trying again.',
  })
  const expectedUrl = `https://github.com/${proposal.repository_full_name}/pull/${data.number}`
  if (!Number.isSafeInteger(data.number) || data.number < 1 || data.url !== expectedUrl
      || data.branch !== `repofix/${proposal.proposal_id}` || typeof data.draft !== 'boolean'
      || !['open', 'closed'].includes(data.state)) {
    throw new WorkbenchApiError('The draft PR could not be confirmed. Check GitHub before trying again.')
  }
  return data
}