/** Exported clustering is independent of the legacy workspace snapshot. */
export function isOrganizationClusteringRoute(
  pathname: string,
  search: string,
) {
  pathname = pathname.replace(/\/+$/, '')
  return (
    pathname === '/organization/clustering' ||
    (pathname === '/organization/explore' &&
      new URLSearchParams(search).get('view') === 'clustering')
  )
}
