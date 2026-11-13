import { useCallback, useEffect, useState } from 'react'
import { currentAccount } from './api/proposalValidation.js'

export default function useGithubSession() {
  const [session, setSession] = useState({ status: 'checking' })
  const [attempt, setAttempt] = useState(0)
  const [signingOut, setSigningOut] = useState(false)
  const [signOutError, setSignOutError] = useState('')

  useEffect(() => {
    const controller = new AbortController()

    currentAccount(controller.signal)
      .then((account) => {
        if (controller.signal.aborted) return

        if (
          !Number.isSafeInteger(account.id) ||
          account.id <= 0 ||
          typeof account.login !== 'string' ||
          !account.login
        ) {
          throw new Error('RepoFix AI returned an invalid GitHub account.')
        }

        setSession({ status: 'signed-in', account })
      })
      .catch((failure) => {
        if (controller.signal.aborted) return

        setSession(
          failure.status === 401
            ? { status: 'signed-out' }
            : { status: 'unavailable', message: failure.message },
        )
      })

    return () => controller.abort()
  }, [attempt])

  const onSessionExpired = useCallback(() => {
    setSignOutError('')
    setSession((previous) =>
      previous.status === 'signed-in'
        ? {
            status: 'signed-out',
            message: 'Your session expired. Sign in again to continue.',
          }
        : previous,
    )
  }, [])

  async function signOut() {
    if (signingOut) return

    setSigningOut(true)
    setSignOutError('')

    try {
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'X-RepoFix-CSRF': '1' },
      })

      if (response.status !== 204) throw new Error()

      const url = new URL(window.location.href)
      url.searchParams.delete('github_login')
      url.hash = ''
      window.history.replaceState(window.history.state, '', url)

      setSession({ status: 'signed-out' })
    } catch {
      setSignOutError('Could not sign out. Please try again.')
    } finally {
      setSigningOut(false)
    }
  }

  function retry() {
    setSession({ status: 'checking' })
    setAttempt((value) => value + 1)
  }

  return { session, retry, onSessionExpired, signOut, signingOut, signOutError }
}