import { useEffect, useRef, useState } from 'react'
import {
  createDraftPullRequest, currentAccount, getProposal, getReview, listProposals, listValidations,
  recordDecision, reviewMatches, startValidation,
} from './api/proposalValidation.js'

const reviews = {
  unreviewed: 'Awaiting review', approved_for_validation: 'Approved for validation', rejected: 'Rejected',
}
const verdicts = {
  verified: 'Patch verified', baseline_not_reproduced: 'Failure did not reproduce',
  patched_tests_failed: 'Patched tests failed', different_tests: 'Different tests collected',
  incomplete: 'Incomplete comparison',
}

function dateLabel(value) {
  const date = new Date(value)
  return value && Number.isFinite(date.valueOf()) ? date.toLocaleString() : 'Unknown time'
}

function Diff({ text }) {
  return <pre className="patch-diff" tabIndex={0} aria-label="Proposed patch diff"><code>
    {text.split('\n').map((line, index) => <span key={index} className={
      line.startsWith('+++') || line.startsWith('---') || line.startsWith('@@') ? 'diff-meta'
        : line.startsWith('+') ? 'diff-added' : line.startsWith('-') ? 'diff-removed' : undefined
    }>{line}</span>)}
  </code></pre>
}

function Result({ run, reviewRevision }) {
  const verified = run.verdict === 'verified'
  const revision = run.admission.binding.review_revision
  return <section className="run-result" aria-label="Selected validation result" aria-live="polite">
    <div className="workspace-row">
      <h3>Test comparison</h3>
      <span className={`status-chip ${verified ? 'status-good' : 'status-neutral'}`}>
        {verdicts[run.verdict] ?? 'Comparison complete'}
      </span>
    </div>
    <div className="result-grid">
      {[['Original code', run.baseline], ['Patched code', run.patched]].map(([label, phase]) => (
        <div className="phase-card" key={label}>
          <h4>{label} <span>{phase.collected} collected</span></h4>
          <dl>
            <div><dt>Passed</dt><dd className="count-good">{phase.passed}</dd></div>
            <div><dt>Failed</dt><dd className={phase.failed ? 'count-bad' : ''}>{phase.failed}</dd></div>
            <div><dt>Skipped</dt><dd>{phase.skipped}</dd></div>
            <div><dt>Errors</dt><dd>{phase.errors}</dd></div>
          </dl>
        </div>
      ))}
    </div>
    {verified && <p>The same tests reproduced a failure on the original code and passed after the patch.</p>}
    <p className="run-meta">Approval revision {revision}{reviewRevision !== revision ? ' · Earlier review' : ''} · {dateLabel(run.admission.closed_at)}</p>
    <p className="run-meta">Validation ID <code>{run.admission.validation_id}</code></p>
  </section>
}

function ProposalPanel({ id, setParentBusy }) {
  const [proposal, setProposal] = useState(null)
  const [review, setReview] = useState(null)
  const [page, setPage] = useState(null)
  const [pageNumber, setPageNumber] = useState(1)
  const [selectedRun, setSelectedRun] = useState(null)
  const [checked, setChecked] = useState(false)
  const [reload, setReload] = useState(0)
  const [reloadHistory, setReloadHistory] = useState(0)
  const [reviewLoading, setReviewLoading] = useState(true)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [operation, setOperation] = useState('')
  const [error, setError] = useState('')
  const [historyError, setHistoryError] = useState('')
  const [publishConsent, setPublishConsent] = useState('')
  const [pullRequest, setPullRequest] = useState(null)
  const [publishError, setPublishError] = useState('')
  const actionInFlight = useRef(false)

  useEffect(() => {
    const controller = new AbortController()
    setReviewLoading(true)
    setError('')
    setReview(null)
    setChecked(false)
    setPublishConsent('')
    Promise.all([getProposal(id, controller.signal), getReview(id, controller.signal)])
      .then(([proposalData, reviewData]) => {
        if (controller.signal.aborted) return
        if (!reviewMatches(proposalData, reviewData) || typeof proposalData.unified_diff !== 'string') {
          throw new Error('The saved proposal and review could not be matched.')
        }
        setProposal(proposalData)
        setReview(reviewData)
      })
      .catch((error) => { if (!controller.signal.aborted) setError(error.message) })
      .finally(() => { if (!controller.signal.aborted) setReviewLoading(false) })
    return () => controller.abort()
  }, [id, reload])

  useEffect(() => {
    const controller = new AbortController()
    setHistoryLoading(true)
    setHistoryError('')
    listValidations(id, pageNumber, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setPage(data)
      })
      .catch((error) => { if (!controller.signal.aborted) setHistoryError(error.message) })
      .finally(() => { if (!controller.signal.aborted) setHistoryLoading(false) })
    return () => controller.abort()
  }, [id, pageNumber, reloadHistory])

  async function act(action) {
    if (actionInFlight.current || !reviewMatches(proposal, review)) return
    if (action === 'approve_for_validation' && !checked) return
    actionInFlight.current = true
    setOperation(action)
    setParentBusy(true)
    setError('')
    setPublishConsent('')
    try {
      if (action === 'validate') {
        const result = await startValidation(proposal, review)
        setSelectedRun(result)
        setPageNumber(1)
        setReloadHistory((value) => value + 1)
      } else {
        setReview(await recordDecision(proposal, review, action))
        setChecked(false)
      }
    } catch (error) {
      setError(error.message)
      // A lost or conflicting write response is not safe to retry blindly.
      if ([0, 401, 409].includes(error.status) || action !== 'validate') {
        setReview(null)
        setChecked(false)
      }
      if (action === 'validate') setReloadHistory((value) => value + 1)
    } finally {
      actionInFlight.current = false
      setOperation('')
      setParentBusy(false)
    }
  }

  const ready = !reviewLoading && reviewMatches(proposal, review)
  const approved = ready && review.status === 'approved_for_validation'
  const recentRuns = page?.results ?? []
  const visibleResult = selectedRun ?? (pageNumber === 1 ? recentRuns[0] : null)
  const validationBinding = visibleResult?.admission.binding
  const eligibleForDraft = approved && visibleResult?.verdict === 'verified'
    && validationBinding.proposal_id === proposal.proposal_id
    && validationBinding.patch_sha256 === proposal.patch_sha256
    && validationBinding.base_commit_sha === proposal.base_commit_sha
    && validationBinding.review_revision === review.revision
    && validationBinding.decision_id === review.current_decision?.decision_id
  const consentKey = eligibleForDraft ? visibleResult.admission.validation_id : ''

  async function publish() {
    if (actionInFlight.current || !eligibleForDraft || publishConsent !== consentKey || pullRequest) return
    actionInFlight.current = true
    setOperation('publish')
    setParentBusy(true)
    setPublishError('')
    setPublishConsent('')
    try {
      setPullRequest(await createDraftPullRequest(proposal, review, visibleResult))
    } catch (error) {
      setPublishError(error.message)
    } finally {
      actionInFlight.current = false
      setOperation('')
      setParentBusy(false)
    }
  }

  return <div className="proposal-panel">
    <div className="workspace-row">
      <div><h3>{proposal?.draft?.title ?? 'Proposal details'}</h3>
        {proposal && <p className="muted">{proposal.repository_full_name} · <code title={proposal.base_commit_sha}>{proposal.base_commit_sha.slice(0, 12)}</code></p>}
      </div>
      <button className="quiet-button" disabled={reviewLoading || !!operation} onClick={() => setReload((value) => value + 1)}>Refresh review</button>
    </div>
    {reviewLoading && <p role="status">Loading saved proposal and review…</p>}
    {error && <p role="alert" className="error-message">{error}</p>}
    {proposal && <details className="patch-review">
      <summary>Review proposed changes <span>{proposal.changes.length} files</span></summary>
      <Diff text={proposal.unified_diff} />
    </details>}
    {ready && <div className="review-actions">
      <span className={`status-chip ${approved ? 'status-good' : 'status-neutral'}`}>{reviews[review.status]} · r{review.revision}</span>
      {!approved && <label className="review-confirmation">
        <input type="checkbox" checked={checked} disabled={!!operation} onChange={(event) => setChecked(event.target.checked)} />
        I reviewed this exact patch.
      </label>}
      <div className="action-row">
        {approved
          ? <button disabled={!!operation} onClick={() => act('validate')}>{operation === 'validate' ? 'Running validation…' : 'Run validation'}</button>
          : <button disabled={!checked || !!operation} onClick={() => act('approve_for_validation')}>Approve for validation</button>}
        {review.status !== 'rejected' && <button className="quiet-button" disabled={!!operation} onClick={() => act('reject')}>
          {approved ? 'Revoke approval' : 'Reject proposal'}
        </button>}
      </div>
      {approved && <p className="muted">Runs the original and patched tests in local Docker.</p>}
    </div>}
    {operation && operation !== 'publish' && <p role="status">{operation === 'validate' ? 'Waiting for both test runs to finish…' : 'Saving review decision…'}</p>}
    {visibleResult && <Result run={visibleResult} reviewRevision={review?.revision ?? visibleResult.admission.binding.review_revision} />}
    {(eligibleForDraft || pullRequest || publishError) && <section className="review-actions" aria-label="GitHub draft pull request">
      <h3>GitHub draft pull request</h3>
      {pullRequest ? <p role="status">
        <a href={pullRequest.url} target="_blank" rel="noreferrer">Open {pullRequest.draft ? 'draft ' : ''}PR #{pullRequest.number} on GitHub</a>
        {pullRequest.state === 'closed' && ' (closed)'}
      </p> : eligibleForDraft && <>
        <p>RepoFix will publish the verified files on a separate branch in <strong>{proposal.repository_full_name}</strong> and open a draft PR.</p>
        <label className="review-confirmation">
          <input type="checkbox" disabled={!!operation} checked={publishConsent === consentKey}
            onChange={(event) => setPublishConsent(event.target.checked ? consentKey : '')} />
          I approve publishing this exact patch to GitHub as a draft PR.
        </label>
        <div className="action-row">
          <button type="button" disabled={!!operation || publishConsent !== consentKey} onClick={publish}>
            {operation === 'publish' ? 'Creating draft PR…' : 'Create draft PR'}
          </button>
        </div>
      </>}
      {publishError && <p role="alert" className="error-message">{publishError}</p>}
    </section>}
    <section className="run-history" aria-label="Validation history">
      <div className="workspace-row">
        <h3>Saved runs</h3>
        <button className="quiet-button" disabled={historyLoading || !!operation} onClick={() => {
          setPageNumber(1); setReloadHistory((value) => value + 1)
        }}>Refresh history</button>
      </div>
      {historyLoading && <p role="status">Loading saved runs…</p>}
      {historyError && <p role="alert" className="error-message">{historyError}</p>}
      {!historyLoading && !historyError && !recentRuns.length && <p className="muted">No saved validations on this page.</p>}
      {!historyLoading && !historyError && recentRuns.length > 0 && <ul className="saved-runs">
        {recentRuns.map((run) => <li key={run.admission.validation_id}>
          <button className="quiet-button saved-run" disabled={!!operation} aria-pressed={visibleResult?.admission.validation_id === run.admission.validation_id} onClick={() => { setPublishConsent(''); setSelectedRun(run) }}>
            <strong>{verdicts[run.verdict] ?? 'Comparison complete'}</strong>
            <span>{dateLabel(run.admission.closed_at)}</span>
          </button>
        </li>)}
      </ul>}
      {(pageNumber > 1 || page?.has_more) && <div className="action-row">
        <button className="quiet-button" disabled={pageNumber === 1 || historyLoading || !!operation} onClick={() => { setPublishConsent(''); setPageNumber((number) => number - 1) }}>Previous runs</button>
        <button className="quiet-button" disabled={!page?.has_more || historyLoading || !!historyError || !!operation} onClick={() => { setPublishConsent(''); setPageNumber(page.next_page) }}>Older runs</button>
      </div>}
    </section>
  </div>
}

export default function ValidationWorkspace({ requestedProposal = null }) {
  const loginOutcome = new URLSearchParams(window.location.search).get('github_login')
  const [account, setAccount] = useState(null)
  const [checking, setChecking] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const [page, setPage] = useState(1)
  const [proposals, setProposals] = useState(null)
  const [selection, setSelection] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [linkedProposal, setLinkedProposal] = useState(requestedProposal)

  useEffect(() => {
    if (!busy) setLinkedProposal(requestedProposal)
  }, [requestedProposal, busy])

  useEffect(() => {
    const controller = new AbortController()
    setChecking(true)
    setError('')
    async function load() {
      try {
        const user = await currentAccount(controller.signal)
        if (controller.signal.aborted) return
        setAccount(user)
        if (linkedProposal) return
        const data = await listProposals(page, controller.signal)
        if (controller.signal.aborted) return
        setProposals(data)
        setSelection((previous) => data.proposals.some((item) => item.proposal_id === previous)
          ? previous : data.proposals[0]?.proposal_id ?? '')
      } catch (error) {
        if (controller.signal.aborted) return
        if (error.status === 401) setAccount(null)
        else setError(error.message)
        setProposals(null)
        setSelection('')
      } finally {
        if (!controller.signal.aborted) setChecking(false)
      }
    }
    load()
    return () => controller.abort()
  }, [page, refresh, linkedProposal])

  return <section className="validation-workspace" aria-labelledby="workspace-heading">
    <div className="workspace-row">
      <div><h2 id="workspace-heading">Validate a saved proposal</h2>
        <p className="muted">Review the diff, approve it, and compare the test results.</p></div>
      {account && <span className="status-chip status-good">GitHub · @{account.login} · ID {account.id}</span>}
    </div>
    {checking && <p role="status">{linkedProposal ? 'Checking GitHub connection…' : 'Checking GitHub connection and saved proposals…'}</p>}
    {error && <p role="alert" className="error-message">{error}</p>}
    {!checking && !account && <div className="connection-card">
      {loginOutcome === 'cancelled' && <p role="status">GitHub sign-in was cancelled. You can try again.</p>}
      {loginOutcome === 'failed' && <p role="alert" className="error-message">Could not connect GitHub. Please try again.</p>}
      <p>Sign in with GitHub to load your saved proposals.</p>
      <div className="action-row">
        <a className="workbench-link-button" href="/api/auth/github/login?flow=workbench">Connect GitHub</a>
      </div>
    </div>}
    {account && linkedProposal && <div className="action-row">
      {busy
        ? <span className="muted">Wait for the current action to finish before switching proposals.</span>
        : <a href="#/validation">Back to saved proposals</a>}
    </div>}
    {account && !linkedProposal && <div className="proposal-picker">
      {proposals?.proposals.length > 0 ? <label htmlFor="saved-proposal">Saved proposal
        <select id="saved-proposal" disabled={checking || busy} value={selection} onChange={(event) => setSelection(event.target.value)}>
          {proposals.proposals.map((proposal) => <option key={proposal.proposal_id} value={proposal.proposal_id}>
            {proposal.title} · {proposal.repository_full_name}
          </option>)}
        </select>
      </label> : !checking && !error && <p className="muted">No saved proposals. Create a fix proposal before starting validation.</p>}
      <div className="action-row">
        <button className="quiet-button" disabled={checking || busy} onClick={() => setRefresh((value) => value + 1)}>Refresh proposals</button>
        {page > 1 && <button className="quiet-button" disabled={checking || busy} onClick={() => setPage(page - 1)}>Previous proposals</button>}
        {proposals?.has_more && <button className="quiet-button" disabled={checking || busy} onClick={() => setPage(proposals.next_page)}>Older proposals</button>}
      </div>
    </div>}
    {!checking && account && (linkedProposal || selection) && <ProposalPanel
      key={`${account.id}:${linkedProposal || selection}`} id={linkedProposal || selection} setParentBusy={setBusy} />}
  </section>
}
