/** Detector evidence has its own contract and never depends on workspace demo mode. */
export function isDetectionRoute(pathname: string, search: string) {
  pathname = pathname.replace(/\/+$/, '')
  return (
    pathname === '/organization/detector' ||
    (pathname === '/organization/evaluation' &&
      !['splits', 'reference'].includes(
        new URLSearchParams(search).get('view') ?? 'detector',
      ))
  )
}
