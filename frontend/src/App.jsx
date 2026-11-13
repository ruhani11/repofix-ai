import LoginPage from './LoginPage.jsx'
import Workbench from './Workbench.jsx'
import useGithubSession from './useGithubSession.js'
import './repofix-workbench.css'

export default function App() {
  const { session, retry, onSessionExpired, signOut, signingOut, signOutError } = useGithubSession()

  if (session.status !== 'signed-in') {
    return <LoginPage status={session.status} message={session.message} onRetry={retry} />
  }

  return <Workbench key={session.account.id} account={session.account} onSessionExpired={onSessionExpired}
    onSignOut={signOut} signingOut={signingOut} signOutError={signOutError} />
}
