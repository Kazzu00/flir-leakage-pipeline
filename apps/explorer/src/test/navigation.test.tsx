import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '@/app/App'
import {
  clusteringFixtureResponse,
  organizationClusteringFixture,
} from './organization-clustering-fixture'

function open(path: string) {
  window.history.replaceState({}, '', path)
  render(<App />)
}

afterEach(() => vi.unstubAllGlobals())

describe('Research exploration flows', () => {
  it('navigates from the complete pipeline to a module', async () => {
    open('/')

    expect(
      await screen.findByRole('heading', {
        name: 'One pipeline. Three perspectives.',
      }),
    ).toBeVisible()

    const links = screen.getAllByRole('link', { name: /Explore module/ })

    expect(links).toHaveLength(3)

    await userEvent.click(links[1])

    expect(
      await screen.findByRole('heading', {
        name: 'Representation & organization',
        level: 1,
      }),
    ).toBeVisible()

    await userEvent.click(screen.getByRole('link', { name: 'Overview' }))

    expect(
      await screen.findByText('What data and analysis are available?'),
    ).toBeVisible()
  })

  it('opens exported clustering and resets the selection when its run changes', async () => {
    const data = organizationClusteringFixture()
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>(async (input) =>
        clusteringFixtureResponse(data, input),
      ),
    )
    open('/organization/clustering')
    const cluster = await screen.findByLabelText('Cluster', { exact: true })
    expect(screen.queryByLabelText('Encoder', { exact: true })).toBeNull()
    await userEvent.selectOptions(cluster, '42')
    const grid = within(screen.getByRole('list', { name: 'Cluster contents' }))
    const thumbnail = grid.getByRole('button', {
      name: 'Inspect content synthetic-content-2',
    })
    await userEvent.click(thumbnail)
    expect(thumbnail).toHaveAttribute('aria-pressed', 'true')
    const selected = within(
      screen
        .getByRole('heading', { name: 'Selected content' })
        .closest('section')!,
    )
    expect(
      selected.getByText('synthetic-content-2', { exact: true }),
    ).toBeVisible()
    await userEvent.selectOptions(
      screen.getByLabelText('Clustering configuration'),
      data.clustering_configurations[1].cluster_run_id,
    )
    expect(screen.getByLabelText('Cluster', { exact: true })).toHaveValue('19')
    expect(
      screen.getByText(
        'Select a content thumbnail to inspect its stored evidence.',
      ),
    ).toBeVisible()
    expect(screen.queryByRole('button', { name: 'AFTER SPLIT' })).toBeNull()
  })

  it('rejects incomplete real split evidence without substituting demo groups', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>(async () => new Response('', { status: 404 })),
    )
    open('/organization/splits')
    await screen.findByText('Visual inspection of real split memberships')
    expect(
      await screen.findByText(
        /UNAVAILABLE: this split family/,
        {},
        { timeout: 5000 },
      ),
    ).toBeVisible()
    expect(screen.queryByRole('button', { name: 'BEFORE SPLIT' })).toBeNull()
    expect(screen.queryByLabelText('Data mode', { exact: true })).toBeNull()
  })

  it('filters experiments and exposes run metadata', async () => {
    open('/experiments')

    await screen.findByRole('heading', {
      name: 'Experiments',
    })

    await userEvent.selectOptions(
      screen.getByLabelText('Module'),
      'segmentation',
    )

    const table = screen.getByRole('table', {
      name: 'Demo experiment registry',
    })

    expect(within(table).getAllByRole('row')).toHaveLength(2)

    await userEvent.click(
      screen.getByRole('button', {
        name: 'demo-panoptic-01',
      }),
    )

    expect(screen.getByText('Artifact references')).toBeVisible()

    await userEvent.type(screen.getByRole('searchbox'), 'does-not-exist')

    expect(screen.getByText('No runs match these filters.')).toBeVisible()
  })
})
