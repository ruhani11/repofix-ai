import './LoginPage.css'
import { useEffect } from 'react'

export default function LoginPage({ status, message, onRetry }) {
  const outcome = new URLSearchParams(window.location.search).get('github_login')

  useEffect(() => {
    document.title = 'Sign in | RepoFix AI'
  }, [])

  return <main className="sign-in-screen">
    <section className="sign-in-card" aria-labelledby="sign-in-title">
      <span className="sign-in-brand">RepoFix AI</span>
      <h1 id="sign-in-title">Sign in to RepoFix AI</h1>
      <p className="muted">Investigate bugs in your GitHub repositories and validate proposed fixes.</p>
      {status === 'checking' && <p className="sign-in-message" role="status">Checking your session…</p>}
      {status === 'unavailable' && <>
        <p className="error-message sign-in-message" role="alert">{message}</p>
        <button className="quiet-button sign-in-action" type="button" onClick={onRetry}>Retry connection</button>
      </>}
      {status === 'signed-out' && <>
        {outcome === 'cancelled' && <p className="sign-in-message" role="status">GitHub sign-in was cancelled. You can try again.</p>}
        {outcome === 'failed' && <p className="error-message sign-in-message" role="alert">GitHub sign-in failed. Please try again.</p>}
        {message && <p className="sign-in-message" role="status">{message}</p>}
        <a className="workbench-link-button sign-in-action" href="/api/auth/github/login?flow=workbench">
          Continue with GitHub
        </a>
      </>}
    </section>
  </main>
}
