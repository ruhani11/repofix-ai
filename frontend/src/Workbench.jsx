import { useEffect, useState, useSyncExternalStore } from 'react'
import { inspectRepository, listInspectorRepositories } from './api/repositories.js'
import InvestigationWorkspace from './InvestigationWorkspace.jsx'
import RepositoryPicker from './RepositoryPicker.jsx'
import Sidebar from './Sidebar.jsx'
import ValidationWorkspace from './ValidationWorkspace.jsx'

const pages = {
  inspector: {
    label: 'GitHub Investigation',
    description: 'Inspect files at a specific commit or investigate a bug in a public repository.',
  },
  validation: {
    label: 'Validation',
    description: 'Review an approved patch, compare test results, and reopen saved runs.',
  },
}

function currentLocation() {
  return window.location.hash
}

function subscribeToPage(onChange) {
  window.addEventListener('hashchange', onChange)
  return () => window.removeEventListener('hashchange', onChange)
}

function RepositoryInspector({ onSessionExpired }) {
  const [repositoryId, setRepositoryId] = useState('')
  const [selectedPathsText, setSelectedPathsText] = useState('')
  const [catalog, setCatalog] = useState(null)
  const [catalogError, setCatalogError] = useState(null)
  const [refreshCount, setRefreshCount] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [investigationBusy, setInvestigationBusy] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    listInspectorRepositories({ signal: controller.signal })
      .then((listing) => { if (!controller.signal.aborted) setCatalog(listing) })
      .catch((failure) => {
        if (controller.signal.aborted) return
        if (failure.status === 401) onSessionExpired()
        else setCatalogError(failure)
      })
    return () => controller.abort()
  }, [refreshCount, onSessionExpired])

  const selectedRepository = catalog?.repositories.find((repo) => String(repo.id) === repositoryId && !repo.disabled)

  function refreshRepositories() {
    setCatalog(null)
    setCatalogError(null)
    setRepositoryId('')
    setResult(null)
    setError('')
    setRefreshCount((count) => count + 1)
  }

  async function handleInspect(event) {
    event.preventDefault()
    if (loading || investigationBusy || !selectedRepository) return

    setError('')
    setResult(null)

    const selectedPaths = selectedPathsText
      .split('\n')
      .map((path) => path.trim())
      .filter(Boolean)

    if (selectedPaths.length > 3) {
      setError('Select at most three files, with one path per line.')
      return
    }

    if (new Set(selectedPaths).size !== selectedPaths.length) {
      setError('Each selected file path must be unique.')
      return
    }

    setLoading(true)

    try {
      const inspection = await inspectRepository(selectedRepository, selectedPaths)
      setResult(inspection)
    } catch (requestError) {
      if (requestError?.status === 401) {
        onSessionExpired()
        return
      }
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Repository inspection failed. Please try again.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="inspector-workspace">
      <section className="inspection-panel" aria-labelledby="inspection-heading">
        <h2 id="inspection-heading">Inspect repository</h2>

        {!catalog && !catalogError && <p role="status">Loading your GitHub repositories…</p>}
        {catalogError && (
          <div className="connection-card">
            <p role="alert" className="error-message">{catalogError.message}</p>
            <div className="action-row"><button type="button" className="quiet-button" onClick={refreshRepositories}>Retry</button></div>
          </div>
        )}
        {catalog && <form onSubmit={handleInspect} aria-busy={loading}>
          <fieldset disabled={loading || investigationBusy}>
            <legend>Repository and selected files</legend>

            <div className="form-field">
              <RepositoryPicker
                repositories={catalog.repositories}
                selectedRepository={selectedRepository}
                disabled={loading || investigationBusy}
                onSelect={(repo) => {
                  setRepositoryId(String(repo.id))
                  setResult(null)
                  setError('')
                }}
              />
              <p id="repository-picker-help" className="field-help">
                @{catalog.accountLogin} · {catalog.repositories.length} {catalog.repositories.length === 1 ? 'repository' : 'repositories'}
                {!catalog.privateAccess && ' · Public only. Private repos require local read access.'}
                {catalog.limitReached && ' · Listing limit reached.'}
              </p>
            </div>

            <div className="form-field">
              <label htmlFor="selected-paths">Selected file paths (optional)</label>
              <textarea
                id="selected-paths"
                rows={4}
                maxLength={3075}
                value={selectedPathsText}
                onChange={(event) => setSelectedPathsText(event.target.value)}
                placeholder={'README.md\nserver/package.json'}
                aria-describedby="selected-paths-help"
              />
              <p id="selected-paths-help" className="field-help">
                One path per line, up to three files. Leave empty to inspect structure only.
              </p>
            </div>

            <div className="action-row">
              <button type="submit" disabled={!selectedRepository}>
                {loading ? 'Inspecting…' : 'Inspect repository'}
              </button>
              <button type="button" className="quiet-button" onClick={refreshRepositories}>Refresh list</button>
            </div>
          </fieldset>
        </form>}

        <p role="status" className="loading-message">
          {loading ? 'Fetching repository metadata, structure, and selected files…' : ''}
        </p>

        {error && <p role="alert" className="error-message">{error}</p>}
      </section>

      {catalog && <InvestigationWorkspace
        key={selectedRepository?.id ?? 'no-repository'}
        repository={selectedRepository}
        busy={loading}
        onBusyChange={setInvestigationBusy}
        onSessionExpired={onSessionExpired}
      />}

      {result && (
        <section className="inspection-results" aria-label="Inspection results">
          <section className="repository-summary">
            <h2>{result.snapshot.repository.full_name}</h2>
            <p>{result.snapshot.repository.description || 'No description provided.'}</p>
            <dl>
              <dt>Default branch</dt>
              <dd>{result.snapshot.repository.default_branch}</dd>
              <dt>Language</dt>
              <dd>{result.snapshot.repository.language || 'Not specified'}</dd>
              <dt>Captured commit</dt>
              <dd><code>{result.snapshot.commit_sha}</code></dd>
              <dt>Tree entries (files and folders)</dt>
              <dd>{result.tree.entries.length}</dd>
              <dt>Selected files read</dt>
              <dd>{result.files.length}</dd>
            </dl>
          </section>

          <details className="repository-tree">
            <summary>Repository structure ({result.tree.entries.length} entries)</summary>
            <div className="table-container">
              <table>
                <caption>Entries at the captured commit</caption>
                <thead>
                  <tr><th scope="col">Path</th><th scope="col">Type</th><th scope="col">Size</th></tr>
                </thead>
                <tbody>
                  {result.tree.entries.map((entry) => (
                    <tr key={entry.path}>
                      <td><code>{entry.path}</code></td>
                      <td>{entry.type}</td>
                      <td>{entry.size == null ? '—' : `${entry.size} bytes`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          <section className="selected-files" aria-labelledby="files-heading">
            <h2 id="files-heading">Selected file contents</h2>
            {result.files.length === 0 && <p>No file content was requested.</p>}

            {result.files.map((file) => (
              <article className="file-card" key={file.path}>
                <h3>{file.path}</h3>
                <p>{file.size_bytes} bytes</p>
                <p>Commit: <code>{file.commit_sha}</code></p>
                <pre className="source-content"><code>{file.content}</code></pre>
              </article>
            ))}
          </section>
        </section>
      )}
    </div>
  )
}

export default function Workbench({ account, onSessionExpired, onSignOut, signingOut, signOutError }) {
  const location = useSyncExternalStore(subscribeToPage, currentLocation)
  const [route, query] = location.split('?')
  const page = route === '#/validation' ? 'validation' : 'inspector'
  const linkedProposal = route === '#/validation' ? new URLSearchParams(query).get('proposal') : null
  const proposalId = /^[a-f0-9]{32}$/.test(linkedProposal ?? '') ? linkedProposal : null
  const [visited, setVisited] = useState(() => new Set([page]))
  const [lastProposalId, setLastProposalId] = useState(proposalId)

  useEffect(() => {
    if (route !== '#/inspector' && route !== '#/validation') window.location.replace('#/inspector')
  }, [route])

  useEffect(() => {
    document.title = `${pages[page].label} | RepoFix AI`
    // Keep a visited page mounted so a navigation click does not discard its
    // form, selected result, or in-flight validation request.
    setVisited((previous) => previous.has(page) ? previous : new Set([...previous, page]))
  }, [page])

  useEffect(() => {
    if (page === 'validation') setLastProposalId(proposalId)
  }, [page, proposalId])

  return (
    <div className="repofix-app">
      <Sidebar pages={pages} activePage={page} onSignOut={onSignOut} signingOut={signingOut} />
      <main className="app-content">
        <header className="app-header">
          <h1>{pages[page].label}</h1>
          <p>{pages[page].description}</p>
        </header>
        {signOutError && <p className="error-message sign-out-error" role="alert">{signOutError}</p>}
        {(page === 'validation' || visited.has('validation')) && (
          <div className="app-page" hidden={page !== 'validation'}>
            <ValidationWorkspace account={account} onSessionExpired={onSessionExpired}
              requestedProposal={page === 'validation' ? proposalId : lastProposalId} />
          </div>
        )}
        {(page === 'inspector' || visited.has('inspector')) && (
          <div className="app-page" hidden={page !== 'inspector'}>
            <RepositoryInspector onSessionExpired={onSessionExpired} />
          </div>
        )}
      </main>
    </div>
  )
}
