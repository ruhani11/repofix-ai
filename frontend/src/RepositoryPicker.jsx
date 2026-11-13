import { useEffect, useId, useRef, useState } from 'react'

const paths = {
  repository: 'M6 3h11a1 1 0 0 1 1 1v16H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Zm-2 14a2 2 0 0 1 2-2h12M8 7h6',
  search: 'M15.5 15.5 20 20m-3-9a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z',
  chevron: 'm6 9 6 6 6-6',
  lock: 'M7 11V8a5 5 0 0 1 10 0v3m-11 0h12a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1Z',
  check: 'm5 12 4 4L19 6',
}

function PickerIcon({ name }) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>
}

function VisibilityBadge({ visibility }) {
  return <span className="repo-picker-badge" data-visibility={visibility}>
    {visibility === 'private' && <PickerIcon name="lock" />}
    {visibility === 'private' ? 'Private' : 'Public'}
  </span>
}

export default function RepositoryPicker({ repositories, selectedRepository, onSelect, disabled }) {
  const id = useId()
  const root = useRef(null)
  const trigger = useRef(null)
  const search = useRef(null)
  const list = useRef(null)
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [visibility, setVisibility] = useState('all')
  const [placement, setPlacement] = useState('below')

  const term = query.trim().toLowerCase()
  const matches = repositories.filter((repo) =>
    (visibility === 'all' || repo.visibility === visibility) && repo.full_name.toLowerCase().includes(term))

  function dismiss({ restoreFocus = false } = {}) {
    setOpen(false)
    if (restoreFocus) trigger.current?.focus()
  }

  function openPicker() {
    if (disabled) return
    const rect = trigger.current.getBoundingClientRect()
    const spaceBelow = window.innerHeight - rect.bottom
    setPlacement(spaceBelow < 300 && rect.top > spaceBelow ? 'above' : 'below')
    setQuery('')
    setVisibility('all')
    setOpen(true)
  }

  useEffect(() => {
    if (!open) return
    search.current?.focus()
    function dismissOutside(event) {
      if (!root.current?.contains(event.target)) setOpen(false)
    }
    document.addEventListener('pointerdown', dismissOutside)
    return () => document.removeEventListener('pointerdown', dismissOutside)
  }, [open])

  useEffect(() => {
    if (list.current) list.current.scrollTop = 0
  }, [query, visibility])

  function handleKeys(event) {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      dismiss({ restoreFocus: true })
      return
    }
    if (event.key === 'Enter' && event.target === search.current) {
      event.preventDefault()
      const first = matches.find((repo) => !repo.disabled)
      if (first) {
        onSelect(first)
        dismiss({ restoreFocus: true })
      }
      return
    }
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
    const options = [...(list.current?.querySelectorAll('button:not(:disabled)') || [])]
    const index = options.indexOf(event.target)
    if (event.target !== search.current && index === -1) return
    event.preventDefault()
    if (event.key === 'ArrowDown') {
      options[index + 1]?.focus()
    } else if (index > 0) {
      options[index - 1]?.focus()
    } else if (index === 0) {
      search.current?.focus()
    } else {
      options.at(-1)?.focus()
    }
  }

  return (
    <div ref={root} className="repo-picker"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) dismiss()
      }}>
      <label htmlFor="repository-picker" id={`${id}-label`}>GitHub repository</label>
      <button ref={trigger} type="button" id="repository-picker" className="repo-picker-trigger"
        disabled={disabled || !repositories.length} aria-expanded={open}
        aria-controls={open ? `${id}-panel` : undefined}
        aria-labelledby={`${id}-label ${id}-value`} aria-describedby="repository-picker-help"
        onKeyDown={(event) => {
          if (!open && event.key === 'ArrowDown') {
            event.preventDefault()
            openPicker()
          }
        }}
        onClick={() => open ? dismiss() : openPicker()}>
        <PickerIcon name="repository" />
        <span className="repo-picker-value" id={`${id}-value`} title={selectedRepository?.full_name}>
          {selectedRepository?.name || (repositories.length ? 'Choose a repository' : 'No repositories found')}
        </span>
        {selectedRepository && <VisibilityBadge visibility={selectedRepository.visibility} />}
        <PickerIcon name="chevron" />
      </button>

      {open && <div id={`${id}-panel`} className="repo-picker-panel" data-placement={placement}
        role="group" aria-label="Choose a GitHub repository" onKeyDown={handleKeys}>
        <div className="repo-picker-search">
          <PickerIcon name="search" />
          <input ref={search} type="search" aria-label="Search repositories" placeholder="Search repositories…"
            autoComplete="off" value={query} onChange={(event) => setQuery(event.target.value)} />
        </div>
        <div className="repo-picker-toolbar">
          <div className="repo-picker-filters" role="group" aria-label="Repository visibility">
            {['all', 'public', 'private'].map((filter) => (
              <button key={filter} type="button" aria-pressed={visibility === filter}
                onClick={() => setVisibility(filter)}>
                {filter[0].toUpperCase() + filter.slice(1)}
              </button>
            ))}
          </div>
          <span className="repo-picker-count" role="status">{matches.length} {matches.length === 1 ? 'repo' : 'repos'}</span>
        </div>
        <ul ref={list} className="repo-picker-options" aria-label="Matching repositories">
          {matches.map((repo) => (
            <li key={repo.id}>
              <button type="button" className="repo-picker-option" aria-pressed={repo.id === selectedRepository?.id}
                title={repo.full_name} disabled={repo.disabled}
                onClick={() => { onSelect(repo); dismiss({ restoreFocus: true }) }}>
                <PickerIcon name={repo.id === selectedRepository?.id ? 'check' : 'repository'} />
                <span className="repo-picker-name">{repo.name}</span>
                <VisibilityBadge visibility={repo.visibility} />
              </button>
            </li>
          ))}
        </ul>
        {!matches.length && <p className="repo-picker-empty">
          {query.trim() ? 'No repositories match your search.' :
            visibility === 'private' ? 'No private repositories available.' : 'No repositories available.'}
        </p>}
      </div>}
    </div>
  )
}
