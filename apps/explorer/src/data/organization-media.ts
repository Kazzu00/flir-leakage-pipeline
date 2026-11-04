export const DEFAULT_ORGANIZATION_MEDIA_BASE_URL =
  '/runtime/organization-media'

function normalizeMediaBaseUrl(baseUrl: string) {
  const normalized = baseUrl.replace(/\/+$/, '')

  if (
    !normalized.startsWith('/') ||
    normalized.startsWith('//') ||
    normalized.includes('\\') ||
    normalized.split('/').includes('..')
  ) {
    throw new Error('invalid-organization-media-base-url')
  }

  return normalized
}

export function organizationMediaUrl(
  relativePath: string,
  baseUrl = DEFAULT_ORGANIZATION_MEDIA_BASE_URL,
) {
  const normalizedBase = normalizeMediaBaseUrl(baseUrl)

  if (
    !relativePath ||
    relativePath.startsWith('/') ||
    relativePath.includes('\\')
  ) {
    throw new Error('invalid-organization-media-relative-path')
  }

  const parts = relativePath.split('/')

  if (
    parts.some(
      (part) =>
        part.length === 0 ||
        part === '.' ||
        part === '..',
    )
  ) {
    throw new Error('invalid-organization-media-relative-path')
  }

  const safePath = parts
    .map((part) => encodeURIComponent(part))
    .join('/')

  return `${normalizedBase}/${safePath}`
}