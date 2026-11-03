import { useEffect, useRef, useState } from 'react'
import { NavLink, Outlet, Link, useLocation } from 'react-router-dom'
import {
  Activity,
  ArrowUpRight,
  Box,
  ChevronRight,
  FlaskConical,
  GitBranch,
  Layers,
  Menu,
  Moon,
  Network,
  PanelLeftClose,
  Sun,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { nav } from '@/app/navigation'
import { useSnapshot, DataModeSwitch } from '@/data/provider'
import { isDetectionRoute } from '@/features/detection/route'
import { isOrganizationClusteringRoute } from '@/features/organization/clustering/route'
import { useDetectionSnapshot } from '@/features/detection/query'
import { useQuery } from '@tanstack/react-query'
import { loadAudit } from '@/data/audit-client'
const icons = [Network, Layers, GitBranch, Box, FlaskConical, Activity]
export function AppShell() {
  const [open, setOpen] = useState(false)
  const [light, setLight] = useState(
    () => localStorage.getItem('flir-theme') === 'light',
  )
  const location = useLocation()
  const audit =
    location.pathname === '/organization/sequences' &&
    new URLSearchParams(location.search).get('mode') === 'audit'
  const auditQuery = useQuery({
    queryKey: ['local-audit'],
    queryFn: loadAudit,
    enabled: audit,
    retry: false,
  })
  const auditMode = auditQuery.isError
    ? 'AUDIT UNAVAILABLE'
    : !auditQuery.data
      ? 'AUDIT LOADING'
      : auditQuery.data.assets.length > 0 &&
          auditQuery.data.assets.every((a) => a.metadata.synthetic === true)
        ? 'DEMO AUDIT'
        : 'LOCAL EXPLORATORY AUDIT'
  const detector = isDetectionRoute(location.pathname, location.search)
  const clustering = isOrganizationClusteringRoute(
    location.pathname,
    location.search,
  )
  const organizationEvidence =
    !detector && /^\/organization(?:\/|$)/.test(location.pathname)
  const { data } = useSnapshot(!detector && !organizationEvidence)
  const mixed = !!data?.leakage
  const detection = useDetectionSnapshot(detector)
  const detectorMode = detection.error
    ? 'UNAVAILABLE'
    : detection.data
      ? 'REAL / VERIFIED'
      : 'LOADING'
  const mainRef = useRef<HTMLElement>(null)
  useEffect(() => {
    const dismiss = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', dismiss)
    return () => window.removeEventListener('keydown', dismiss)
  }, [])
  useEffect(() => {
    document.documentElement.dataset.theme = light ? 'light' : 'dark'
    localStorage.setItem('flir-theme', light ? 'light' : 'dark')
  }, [light])
  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true })
    document.title = `${location.pathname === '/' ? 'Pipeline overview' : location.pathname.split('/').filter(Boolean).join(' / ')} · FLIR Explorer`
  }, [location.pathname])
  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <aside className={`sidebar ${open ? 'is-open' : ''}`}>
        <Link to="/" className="brand" onClick={() => setOpen(false)}>
          <span className="brand-symbol">
            <Network size={23} />
          </span>
          <span>
            FLIR<span className="brand-sub">PIPELINE EXPLORER</span>
          </span>
        </Link>
        <div className="workspace">
          <span className="workspace-dot" />
          Amazonia research<span className="micro">v0.1</span>
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav aria-label="Main navigation">
          {nav.map(([path, label], i) => {
            const Icon = icons[i]
            return (
              <NavLink
                key={path}
                to={path}
                end={path === '/'}
                onClick={() => setOpen(false)}
              >
                <Icon size={17} />
                <span>{label}</span>
                {i > 0 && i < 4 && <small>0{i}</small>}
              </NavLink>
            )
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="side-note">
            <span className="status status-mock">
              {audit
                ? auditMode
                : organizationEvidence
                  ? 'M02 REAL EVIDENCE'
                  : detector
                    ? 'M02 DETECTOR CONTRACT'
                    : mixed
                      ? 'MIXED WORKSPACE'
                      : '◈ DEMO WORKSPACE'}
            </span>
            <p>
              {audit
                ? 'audit-evidence-v1 · local evidence and human decisions.'
                : clustering
                  ? 'organization-evidence-v2 · stored cluster evidence.'
                  : organizationEvidence
                    ? 'organization-evidence-v2 · stored evidence; see resource status.'
                    : detector
                      ? 'Other modules retain their own data modes.'
                      : mixed
                        ? 'M02 local artifacts.'
                        : 'Synthetic data.'}
              <br />
              Real research boundaries.
            </p>
          </div>
          <a
            href="https://github.com/Kazzu00/flir-pipeline-explorer"
            target="_blank"
            rel="noreferrer"
          >
            Project repository <ArrowUpRight size={14} />
          </a>
          <span className="micro muted">Three pipelines. One perspective.</span>
        </div>
      </aside>
      {open && (
        <button
          aria-label="Close navigation"
          className="nav-backdrop"
          onClick={() => setOpen(false)}
        />
      )}
      <div className="main-shell">
        <header className="topbar">
          <Button
            className="menu-button"
            variant="ghost"
            size="icon"
            aria-label={open ? 'Hide navigation' : 'Open navigation'}
            aria-expanded={open}
            onClick={() => setOpen(!open)}
          >
            {open ? <PanelLeftClose /> : <Menu />}
          </Button>
          <nav aria-label="Breadcrumb" className="breadcrumb">
            <Link to="/">Workspace</Link>
            <ChevronRight size={13} />
            <span>
              {location.pathname === '/'
                ? 'Pipeline overview'
                : location.pathname.split('/').filter(Boolean).join(' / ')}
            </span>
          </nav>
          <div className="header-actions">
            {!detector && !organizationEvidence && <DataModeSwitch />}
            <span className="local-status">
              <span />
              {audit
                ? auditMode
                : clustering
                  ? 'M02 real contracts - presentation only'
                  : organizationEvidence
                    ? 'M02 real contracts - presentation only'
                    : detector
                      ? 'Detector export · presentation only'
                      : mixed
                        ? 'Artifact + mock adapters · local'
                        : 'Mock adapter · local'}
            </span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setLight(!light)}
              aria-label={
                light ? 'Switch to dark theme' : 'Switch to light theme'
              }
            >
              {light ? <Moon size={17} /> : <Sun size={17} />}
            </Button>
          </div>
        </header>
        <div className="demo-bar">
          <span>
            {audit
              ? auditMode
              : clustering
                ? 'EXPORTED EVIDENCE'
                : organizationEvidence
                  ? 'EXPORTED EVIDENCE'
                  : detector
                    ? detectorMode
                    : mixed
                      ? 'MIXED'
                      : 'DEMO'}
          </span>
          {audit
            ? 'Local review workspace. Synthetic evidence is labeled DEMO. Import integrity and human review do not establish scientific independence.'
            : clustering
              ? 'Module 02 clustering uses organization-evidence-v2. Export and membership checks do not establish sequence identity or ground truth.'
              : organizationEvidence
                ? 'Module 02 uses real exports only. Missing resources remain unavailable. Candidate evidence is not ground truth; other modules retain their own modes.'
                : detector
                  ? 'Detector contract only. Scientific computation and verification upstream. Modules 01 and 03 remain DEMO.'
                  : mixed
                    ? 'Module 02: local artifacts. Modules 01 and 03: DEMO. Missing stages remain pending.'
                    : 'All displayed collections, runs and numeric metrics are synthetic. Repository evidence is labeled separately.'}
        </div>
        <main id="main" tabIndex={-1} ref={mainRef}>
          <Outlet />
        </main>
        <footer>
          FLIR RESEARCH WORKSPACE{' '}
          <span>Visualization only · no scientific processing</span>
        </footer>
      </div>
    </div>
  )
}
