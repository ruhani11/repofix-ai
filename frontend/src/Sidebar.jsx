import { useRef, useState } from 'react'
import './Sidebar.css'

const icons = {
  validation: 'M9 12l2 2 4-4m5 2a8 8 0 1 1-16 0 8 8 0 0 1 16 0Z',
  inspector: 'M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z',
}

export default function Sidebar({ pages, activePage, onSignOut, signingOut }) {
  const drawer = useRef(null)
  const [open, setOpen] = useState(false)

  function openMenu() {
    if (!drawer.current.open) {
      drawer.current.showModal()
      setOpen(true)
    }
  }

  function closeMenu() {
    if (drawer.current.open) drawer.current.close()
  }

  function signOutFromMenu() {
    closeMenu()
    onSignOut()
  }

  return (
    <>
      <header className="rf-navigation-toolbar">
        <div className="rf-navigation-toolbar-inner">
          <button className="rf-navigation-icon-button" type="button" onClick={openMenu}
            aria-label="Open navigation menu" aria-controls="rf-navigation-drawer" aria-expanded={open}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <a className="rf-navigation-brand" href="#/inspector">RepoFix AI</a>
        </div>
      </header>
      <dialog id="rf-navigation-drawer" ref={drawer} aria-labelledby="rf-navigation-title"
        onClose={() => setOpen(Boolean(drawer.current?.open))}
        onClick={(event) => { if (event.target === event.currentTarget) closeMenu() }}>
        <div className="rf-navigation-panel">
          <div className="rf-navigation-heading">
            <h2 id="rf-navigation-title">Navigation</h2>
            <button className="rf-navigation-icon-button" type="button" onClick={closeMenu} aria-label="Close navigation menu">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" aria-hidden="true">
                <path d="m6 6 12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
          <nav className="rf-navigation-links" aria-label="Workbench pages">
            {Object.entries(pages).map(([id, { label }]) => (
              <a key={id} href={`#/${id}`} onClick={closeMenu} aria-current={activePage === id ? 'page' : undefined}>
                {icons[id] && (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                    <path d={icons[id]} />
                  </svg>
                )}
                {label}
              </a>
            ))}
            <button className="quiet-button" type="button" onClick={signOutFromMenu} disabled={signingOut}>
              {signingOut ? 'Signing out…' : 'Sign out'}
            </button>
          </nav>
        </div>
      </dialog>
    </>
  )
}
