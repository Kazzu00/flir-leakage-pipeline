import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { OrganizationEvidenceProvider } from '@/data/organization-provider'
import { Clustering } from '@/features/organization/clustering/Clustering'
import {
  clusteringFixtureResponse,
  organizationClusteringFixture,
} from './organization-clustering-fixture'

afterEach(() => vi.unstubAllGlobals())

function mount(
  data = organizationClusteringFixture(),
  override?: (input: RequestInfo | URL) => Promise<Response> | undefined,
) {
  const fetchMock = vi.fn<typeof fetch>(
    async (input) =>
      override?.(input) ?? clusteringFixtureResponse(data, input),
  )
  vi.stubGlobal('fetch', fetchMock)
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <OrganizationEvidenceProvider>
        <Clustering />
      </OrganizationEvidenceProvider>
    </QueryClientProvider>,
  )
  return fetchMock
}

function tiles() {
  return within(
    screen.getByRole('list', { name: 'Cluster contents' }),
  ).getAllByRole('button')
}
function detail() {
  return within(
    screen
      .getByRole('heading', { name: 'Selected content' })
      .closest('section')!,
  )
}
function fact(label: string) {
  return detail().getByText(label, { selector: 'dt', exact: true })
    .nextElementSibling!
}

describe('Organization clustering contract view (synthetic inputs)', () => {
  it('labels preserved membership from exported strategies without inferring missing metadata', async () => {
    const data = organizationClusteringFixture()
    const frozen = data.clustering_configurations[1]
    frozen.strategy_labels = ['SAMPLE-A', 'SAMPLE-B']
    frozen.membership_consistency_verified = true
    frozen.noise_fraction = 0.123456
    mount(data)
    const selector = await screen.findByLabelText('Clustering configuration')
    await userEvent.selectOptions(selector, frozen.cluster_run_id)
    expect(
      within(selector).getByRole('option', { selected: true }),
    ).toHaveTextContent(
      'SAMPLE-A, SAMPLE-B · preserved membership · 1 cluster · 12.35% noise',
    )
    expect(selector).not.toHaveTextContent(frozen.cluster_run_id)
    const overview = screen
      .getByRole('heading', { name: 'Configuration overview' })
      .closest('section')!
    const facts = within(overview)
    const value = (label: string) =>
      facts.getByText(label, { selector: 'dt', exact: true })
        .nextElementSibling!
    expect(value('Strategy label')).toHaveTextContent('SAMPLE-A, SAMPLE-B')
    for (const label of ['Algorithm', 'Representation', 'Extractor'])
      expect(value(label)).toHaveTextContent('Not preserved in export')
    expect(value('Original clustering artifact')).toHaveTextContent(
      'Not included in export',
    )
    expect(value('Stored membership consistency')).toHaveTextContent(
      'Verified in export',
    )
    expect(value('Source kind')).toHaveTextContent('Preserved split membership')
    expect(overview).not.toHaveTextContent(
      /DBSCAN|DINOv2|CLIP|optimal|winner|recommended/i,
    )
    expect(overview).toHaveTextContent('it does not establish ground truth')
    await userEvent.click(
      screen.getByRole('button', { name: 'Configuration provenance' }),
    )
    const dialog = screen.getByRole('dialog', {
      name: 'Configuration provenance',
    })
    expect(JSON.parse(dialog.querySelector('pre')!.textContent!)).toEqual(
      frozen,
    )
    expect(dialog).toHaveTextContent('frozen_split_membership')
    expect(dialog).toHaveTextContent(frozen.cluster_run_id)
  })

  it('shows reported full-artifact metadata and uses a human fallback for unlabeled frozen membership', async () => {
    const data = organizationClusteringFixture()
    mount(data)
    const selector = await screen.findByLabelText('Clustering configuration')
    const overview = screen
      .getByRole('heading', { name: 'Configuration overview' })
      .closest('section')!
    const facts = within(overview)
    expect(
      facts.getByText('Algorithm', { selector: 'dt' }).nextElementSibling,
    ).toHaveTextContent('DBSCAN')
    expect(
      facts.getByText('Representation', { selector: 'dt' }).nextElementSibling,
    ).toHaveTextContent(data.clustering_configurations[0].representation!)
    expect(
      facts.getByText('Extractor', { selector: 'dt' }).nextElementSibling,
    ).toHaveTextContent(data.clustering_configurations[0].extractor!)
    expect(facts.getByText('Included in export', { exact: true })).toBeVisible()
    expect(
      within(selector).getByRole('option', { selected: true }),
    ).toHaveTextContent('DBSCAN')
    await userEvent.selectOptions(
      selector,
      data.clustering_configurations[1].cluster_run_id,
    )
    expect(
      within(selector).getByRole('option', { selected: true }),
    ).toHaveTextContent('Preserved split membership · 1 cluster · 0.00% noise')
    expect(selector).not.toHaveTextContent(/null|undefined|synthetic-run/)
  })

  it('switches density without resetting selection, batches, previews or resources', async () => {
    const fetchMock = mount(organizationClusteringFixture(125))
    const grid = await screen.findByRole('list', { name: 'Cluster contents' })
    const density = screen.getByLabelText('Density', { exact: true })
    expect(density).toHaveValue('compact')
    expect(grid).toHaveAttribute('data-density', 'compact')
    expect(tiles()[0]).toHaveAccessibleName(
      'Inspect content synthetic-content-0',
    )
    expect(tiles()[0]).toHaveAccessibleDescription(
      'SAMPLE frame 0.jpg, Frame 0',
    )
    expect(tiles()[0].querySelector('.cluster-tile-text')).toBeNull()
    const preview = within(tiles()[0]).getByRole('img')
    await userEvent.click(tiles()[0])
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }))
    expect(tiles()).toHaveLength(120)
    const requests = fetchMock.mock.calls.length
    await userEvent.selectOptions(density, 'detailed')
    expect(grid).toHaveAttribute('data-density', 'detailed')
    expect(within(tiles()[0]).getByText('SAMPLE frame 0.jpg')).toBeVisible()
    expect(within(tiles()[0]).getByText('Frame 0')).toBeVisible()
    expect(
      within(tiles()[0]).getByText('Source video: synthetic-source'),
    ).toBeVisible()
    expect(tiles()[0]).toHaveAttribute('aria-pressed', 'true')
    expect(fact('Content ID')).toHaveTextContent('synthetic-content-0')
    expect(within(tiles()[0]).getByRole('img')).toBe(preview)
    fireEvent.error(preview)
    await userEvent.selectOptions(density, 'compact')
    expect(tiles()).toHaveLength(120)
    expect(tiles()[0]).toHaveAttribute('aria-pressed', 'true')
    expect(
      within(tiles()[0]).getByText('The image could not be loaded.'),
    ).toBeVisible()
    expect(fetchMock).toHaveBeenCalledTimes(requests)
    expect(screen.getByLabelText('Cluster', { exact: true })).toHaveValue('7')
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }))
    expect(tiles()).toHaveLength(125)
  })

  it('loads through the organization provider without a snapshot or invented scatter', async () => {
    const data = organizationClusteringFixture()
    let resolve!: (response: Response) => void
    const pending = new Promise<Response>((done) => {
      resolve = done
    })
    const fetchMock = mount(data, (input) =>
      String(input).endsWith('/clustering_configurations.json')
        ? pending
        : undefined,
    )
    expect(screen.getByRole('status')).toHaveTextContent(
      'Loading clustering configurations',
    )
    await act(async () => {
      resolve(
        clusteringFixtureResponse(data, '/clustering_configurations.json'),
      )
    })
    await screen.findByRole('list', { name: 'Cluster contents' })
    expect(screen.getByLabelText('Clustering configuration')).toHaveValue(
      data.clustering_configurations[0].cluster_run_id,
    )
    expect(tiles()).toHaveLength(2)
    expect(
      fetchMock.mock.calls.every(([input]) =>
        String(input).startsWith('/runtime/organization-evidence-v2/'),
      ),
    ).toBe(true)
    const region = screen.getByRole('region', { name: 'Clustering evidence' })
    expect(region).not.toHaveTextContent(
      /DEMO|SYNTHETIC IDENTITIES|Status mock/,
    )
    expect(region.querySelector('canvas')).toBeNull()
    expect(region).toHaveTextContent(
      'Algorithmic visual grouping · not sequence identity or ground truth.',
    )
    expect(region).toHaveTextContent(
      'Cluster membership does not establish confirmed leakage or temporal identity.',
    )
  })

  it('filters nonconsecutive clusters and puts explicit noise last', async () => {
    mount()
    const selector = await screen.findByLabelText('Cluster', { exact: true })
    expect(
      within(selector)
        .getAllByRole('option')
        .map((option) => option.textContent),
    ).toEqual([
      'Cluster 7 · 2 contents',
      'Cluster 42 · 1 contents',
      'Noise −1 · 1 contents',
    ])
    await userEvent.selectOptions(selector, '42')
    expect(tiles()).toHaveLength(1)
    expect(tiles()[0]).toHaveAccessibleName(
      'Inspect content synthetic-content-2',
    )
    await userEvent.selectOptions(selector, '-1')
    expect(tiles()[0]).toHaveAccessibleName(
      'Inspect content synthetic-content-3',
    )
    expect(
      screen.getByText(/Noise −1 contains unassigned contents/),
    ).toBeVisible()
    await userEvent.click(tiles()[0])
    expect(fact('Is noise')).toHaveTextContent('Yes')
  })

  it('joins media by preview key, handles unavailable previews and load errors', async () => {
    const data = organizationClusteringFixture()
    data.media.reverse()
    mount(data)
    await screen.findByRole('list', { name: 'Cluster contents' })
    const preview = within(tiles()[0]).getByRole('img')
    expect(preview).toHaveAttribute(
      'src',
      '/runtime/organization-media/synthetic/SAMPLE%20frame%200.jpg',
    )
    expect(preview).toHaveAttribute('loading', 'lazy')
    expect(preview).toHaveAttribute('decoding', 'async')
    expect(within(tiles()[1]).getByText('Preview unavailable')).toBeVisible()
    fireEvent.error(preview)
    expect(
      within(tiles()[0]).getByText('The image could not be loaded.'),
    ).toBeVisible()
    await userEvent.click(tiles()[1])
    expect(fact('Content ID')).toHaveTextContent('synthetic-content-1')
  })

  it('selects content by keyboard and preserves multiple occurrences, nulls and zero diagnostics', async () => {
    mount()
    await screen.findByRole('list', { name: 'Cluster contents' })
    tiles()[0].focus()
    await userEvent.keyboard('{Enter}')
    expect(tiles()[0]).toHaveAttribute('aria-pressed', 'true')
    expect(fact('Content ID')).toHaveTextContent('synthetic-content-0')
    expect(fact('Timestamp (seconds)')).toHaveTextContent('0')
    expect(fact('Record count')).toHaveTextContent('2')
    expect(fact('Core distance')).toHaveTextContent('∞ (reported)')
    expect(fact('Probability')).toHaveTextContent('0')
    expect(fact('Reachability')).toHaveTextContent('0')
    expect(fact('Ordering position')).toHaveTextContent('0')
    expect(fact('Annotation consensus')).toHaveTextContent(
      'different label bytes',
    )
    const occurrences = within(
      screen.getByRole('region', { name: 'Content occurrences' }),
    )
    await occurrences.findByText('synthetic-occurrence-copy')
    expect(occurrences.getAllByRole('listitem')).toHaveLength(2)
    expect(occurrences.getAllByText('synthetic-timeline')).toHaveLength(2)
    expect(tiles()).toHaveLength(2)
    await userEvent.click(tiles()[1])
    expect(detail().queryByText('Probability')).toBeNull()
    expect(detail().queryByText('Timestamp (seconds)')).toBeNull()
  })

  it('clears incompatible selection on run changes and describes frozen membership accurately', async () => {
    const data = organizationClusteringFixture()
    mount(data)
    await screen.findByRole('list', { name: 'Cluster contents' })
    await userEvent.click(tiles()[0])
    await userEvent.selectOptions(
      screen.getByLabelText('Clustering configuration'),
      data.clustering_configurations[1].cluster_run_id,
    )
    expect(screen.getByLabelText('Cluster', { exact: true })).toHaveValue('19')
    expect(
      detail().getByText(
        'Select a content thumbnail to inspect its stored evidence.',
      ),
    ).toBeVisible()
    expect(tiles()[0]).toHaveAttribute('aria-pressed', 'false')
    expect(
      screen.getByText(/Preserved membership from frozen splits/),
    ).toBeVisible()
    expect(
      screen.getByText(
        /The original clustering artifact is not included in this export/,
      ),
    ).toBeVisible()
    expect(screen.getByText('Not verified in export')).toBeVisible()
    expect(
      screen.queryByText('Included in export', { exact: true }),
    ).toBeNull()
    expect(
      screen.getByLabelText('Clustering configuration'),
    ).not.toHaveTextContent(/null|undefined/)
  })

  it('renders 60 items per batch and resets the limit and selection on cluster/run changes', async () => {
    const data = organizationClusteringFixture(125)
    mount(data)
    await screen.findByRole('list', { name: 'Cluster contents' })
    expect(tiles()).toHaveLength(60)
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }))
    expect(tiles()).toHaveLength(120)
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }))
    expect(tiles()).toHaveLength(125)
    expect(screen.queryByRole('button', { name: 'Load more' })).toBeNull()
    await userEvent.click(tiles()[80])
    await userEvent.selectOptions(
      screen.getByLabelText('Cluster', { exact: true }),
      '42',
    )
    expect(detail().queryByText('synthetic-content-80')).toBeNull()
    await userEvent.selectOptions(
      screen.getByLabelText('Cluster', { exact: true }),
      '7',
    )
    expect(tiles()).toHaveLength(60)
    await userEvent.click(screen.getByRole('button', { name: 'Load more' }))
    await userEvent.selectOptions(
      screen.getByLabelText('Clustering configuration'),
      data.clustering_configurations[1].cluster_run_id,
    )
    await userEvent.selectOptions(
      screen.getByLabelText('Clustering configuration'),
      data.clustering_configurations[0].cluster_run_id,
    )
    expect(tiles()).toHaveLength(60)
  })

  it.each([
    'clustering_configurations',
    'clusters',
    'cluster_memberships',
    'contents',
  ])('reports invalid %s without fallback', async (resource) => {
    mount(undefined, (input) =>
      String(input).endsWith(`/${resource}.json`)
        ? Promise.resolve(new Response('{}'))
        : undefined,
    )
    expect(
      await screen.findByText(
        /could not be loaded or validated from organization-evidence-v2/,
      ),
    ).toBeVisible()
    expect(screen.queryByRole('list', { name: 'Cluster contents' })).toBeNull()
  })

  it.each(['media', 'records'])(
    'keeps evidence available when %s fails',
    async (resource) => {
      mount(undefined, (input) =>
        String(input).endsWith(`/${resource}.json`)
          ? Promise.resolve(new Response('', { status: 404 }))
          : undefined,
      )
      await screen.findByRole('list', { name: 'Cluster contents' })
      await userEvent.click(tiles()[0])
      expect(
        await screen.findByText(
          resource === 'media'
            ? /Preview metadata could not be loaded/
            : /Occurrence records could not be loaded/,
        ),
      ).toBeVisible()
      expect(fact('Content ID')).toHaveTextContent('synthetic-content-0')
      expect(tiles()).toHaveLength(2)
    },
  )

  it.each(['configurations', 'clusters', 'memberships'] as const)(
    'handles empty %s',
    async (empty) => {
      const data = organizationClusteringFixture()
      if (empty === 'configurations') data.clustering_configurations = []
      else if (empty === 'clusters') {
        data.clusters = []
        data.cluster_memberships = []
      } else data.cluster_memberships = []
      mount(data)
      const message =
        empty === 'configurations'
          ? /No clustering configurations are available/
          : empty === 'clusters'
            ? /No clusters are exported/
            : /No content memberships are exported/
      expect(await screen.findByText(message)).toBeVisible()
    },
  )

  it('rejects duplicate run membership identities rather than duplicating thumbnails', async () => {
    const data = organizationClusteringFixture()
    data.cluster_memberships.push(data.cluster_memberships[0])
    mount(data)
    expect(
      await screen.findByText(/More than one membership row for a content/),
    ).toBeVisible()
    expect(screen.queryByRole('list', { name: 'Cluster contents' })).toBeNull()
  })
})
