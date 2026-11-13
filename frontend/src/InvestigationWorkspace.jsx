import { useRef, useState } from 'react'
import { generateFixProposal, runInvestigation } from './api/proposalValidation.js'

const idPattern = /^[a-f0-9]{32}$/

function Report({ run }) {
  const { report } = run
  const likely = report.status === 'likely_cause'

  return <section className="investigation-report" aria-label="Investigation report">
    <div className="workspace-row">
      <div>
        <h3>Investigation report</h3>
        <p className="muted">{run.repository_full_name} · <code title={run.commit_sha}>{run.commit_sha.slice(0, 12)}</code></p>
      </div>
      <span className="status-chip status-neutral">{likely ? 'Likely cause' : 'Insufficient evidence'}</span>
    </div>
    <p>{report.issue_summary}</p>
    {likely && <div><h4>Likely cause</h4><p>{report.root_cause}</p></div>}
    {report.evidence.length > 0 && <div>
      <h4>Source evidence</h4>
      {report.evidence.map((item, index) => <details className="investigation-evidence" key={index}>
        <summary><code>{item.path}:{item.start_line}{item.end_line !== item.start_line && `–${item.end_line}`}</code></summary>
        <p>{item.explanation}</p>
        <pre className="source-content" tabIndex={0}><code>{item.quote}</code></pre>
      </details>)}
    </div>}
    <div><h4>Next steps</h4><ul>{report.next_steps.map((step, index) => <li key={index}>{step}</li>)}</ul></div>
    {report.limitations.length > 0 && <div><h4>Limitations</h4>
      <ul>{report.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul>
    </div>}
    {!run.history_saved && <p role="status">The report is shown here, but its save was not confirmed. Fix generation requires a saved report.</p>}
  </section>
}

export default function InvestigationWorkspace({ repository, busy, onBusyChange, onSessionExpired }) {
  const [description, setDescription] = useState('')
  const [run, setRun] = useState(null)
  const [proposal, setProposal] = useState(null)
  const [operation, setOperation] = useState('')
  const [error, setError] = useState('')
  const inFlight = useRef(false)

  const isPublic = repository?.visibility === 'public'
  const canGenerate = run?.history_saved === true && run.report.status === 'likely_cause'
  const disabled = busy || !!operation

  async function act(action) {
    if (inFlight.current || busy || !isPublic) return
    const issue = description.trim()
    if (action === 'investigate' && !issue) return
    if (action === 'propose' && (!canGenerate || proposal)) return

    inFlight.current = true
    setOperation(action)
    onBusyChange(true)
    setError('')
    try {
      if (action === 'investigate') {
        setRun(null)
        setProposal(null)
        const result = await runInvestigation(repository, issue)
        if (!idPattern.test(result.run_id ?? '') || result.status !== 'completed'
          || result.repository_full_name?.toLowerCase() !== repository.full_name.toLowerCase()
          || typeof result.commit_sha !== 'string'
          || !['likely_cause', 'insufficient_evidence'].includes(result.report?.status)
          || !Array.isArray(result.report.evidence) || !Array.isArray(result.report.next_steps)
          || !Array.isArray(result.report.limitations)) {
          throw new Error('The investigation response could not be matched to this repository.')
        }
        setRun(result)
      } else {
        const result = await generateFixProposal(run.run_id)
        if (!idPattern.test(result.proposal_id ?? '') || result.status !== 'proposed'
          || result.investigation_run_id !== run.run_id || result.base_commit_sha !== run.commit_sha
          || result.repository_id !== repository.id) {
          throw new Error('The proposal could not be matched to this investigation. Check saved proposals in Validation.')
        }
        setProposal(result)
      }
    } catch (failure) {
      if (failure.status === 401) onSessionExpired(failure)
      else setError(failure.message)
    } finally {
      inFlight.current = false
      setOperation('')
      onBusyChange(false)
    }
  }

  return <section className="investigation-workspace" aria-labelledby="investigation-heading">
    <h2 id="investigation-heading">Investigate an issue</h2>
    {!repository && <p className="muted">Choose a public repository above to start.</p>}
    {repository && !isPublic && <p className="muted">This repository is available for inspection. AI investigations currently support public repositories.</p>}
    {isPublic && <>
      <form aria-busy={!!operation} onSubmit={(event) => { event.preventDefault(); void act('investigate') }}>
        <div className="form-field">
          <label htmlFor="issue-description">What’s going wrong?</label>
          <textarea id="issue-description" rows={5} maxLength={8000} required disabled={disabled}
            value={description} onChange={(event) => setDescription(event.target.value)}
            aria-describedby="investigation-provider-note"
            placeholder="Describe the steps to reproduce, what actually happens, and what you expected." />
          <p id="investigation-provider-note" className="field-help">Investigation and proposal generation send issue text and relevant code to the configured AI provider and may use API credits. Leave out secrets.</p>
        </div>
        <button type="submit" disabled={disabled || !description.trim()}>
          {operation === 'investigate' ? 'Investigating…' : 'Run investigation'}
        </button>
      </form>
      {operation && <p role="status">{operation === 'investigate' ? 'Reading the repository and checking source citations…' : 'Generating a patch and regression tests…'}</p>}
      {error && <p role="alert" className="error-message">{error}</p>}
      {run && <Report run={run} />}
      {canGenerate && !proposal && <div className="action-row">
        <button type="button" disabled={disabled} onClick={() => act('propose')}>Generate fix proposal</button>
      </div>}
      {proposal && <div className="investigation-proposal">
        <h3>{proposal.draft?.title ?? 'Fix proposal'}</h3>
        {proposal.proposal_saved === true
          ? <div className="action-row"><a className="workbench-link-button" href={`#/validation?proposal=${proposal.proposal_id}`}>Review proposal in Validation</a></div>
          : <>
            <p role="status">The draft was returned, but its save was not confirmed. Refresh saved proposals in Validation before generating another.</p>
            {typeof proposal.unified_diff === 'string' && <details className="patch-review"><summary>View returned diff</summary>
              <pre className="patch-diff" tabIndex={0}><code>{proposal.unified_diff}</code></pre>
            </details>}
          </>}
      </div>}
    </>}
  </section>
}
