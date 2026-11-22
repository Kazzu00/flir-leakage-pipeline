import {
  contractFiles,
  payloadNames,
  DetectionManifestSchema,
  DetectionExportSchema,
} from './schema.ts'

export class DetectionContractError extends Error {
  kind: 'missing' | 'validation'
  constructor(kind: 'missing' | 'validation', detail: string) {
    super(detail)
    this.kind = kind
  }
}
/** Exact byte hashes bind the files before any JSON is exposed to presentation. */
export async function validateDetectionFiles(
  files: Record<string, Uint8Array>,
) {
  try {
    const parse = (file: string): unknown =>
      JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(files[file]))
    const manifest = DetectionManifestSchema.parse(parse('manifest.json'))
    await Promise.all(
      contractFiles
        .filter((f) => f !== 'manifest.json')
        .map(async (file) => {
          if (!files[file]) throw new Error(`Missing file: ${file}`)
          const digest = await crypto.subtle.digest(
            'SHA-256',
            new Uint8Array(files[file]),
          )
          const hash = Array.from(new Uint8Array(digest), (b) =>
            b.toString(16).padStart(2, '0'),
          ).join('')
          if (hash !== manifest.file_sha256[file])
            throw new Error(`Checksum mismatch: ${file}`)
        }),
    )
    return DetectionExportSchema.parse(
      Object.fromEntries(
        payloadNames.map((name) => [name, parse(`${name}.json`)]),
      ),
    )
  } catch (error) {
    throw new DetectionContractError(
      'validation',
      error instanceof Error ? error.message : 'Invalid contract',
    )
  }
}
export async function loadDetectionExport(
  urls: Record<string, string>,
  signal?: AbortSignal,
  fetcher: typeof fetch = fetch,
) {
  const entries = await Promise.all(
    contractFiles.map(async (file) => {
      if (!urls[file])
        throw new DetectionContractError(
          'missing',
          `Missing contract URL: ${file}`,
        )
      let response: Response
      try {
        response = await fetcher(urls[file], { signal })
      } catch (error) {
        if (signal?.aborted) throw error
        throw new DetectionContractError('missing', `Could not load ${file}`)
      }
      if (!response.ok)
        throw new DetectionContractError(
          'missing',
          `HTTP ${response.status}: ${file}`,
        )
      return [file, new Uint8Array(await response.arrayBuffer())] as const
    }),
  )
  return validateDetectionFiles(Object.fromEntries(entries))
}
