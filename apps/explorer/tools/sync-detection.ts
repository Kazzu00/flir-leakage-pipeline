import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'
import { contractFiles } from '../src/features/detection/schema.ts'
import { validateDetectionFiles } from '../src/features/detection/adapter.ts'

/** Allowlisted, byte-preserving import. No upstream code or scientific artifact is executed. */
export async function syncDetection(
  source: string,
  destination: string,
  publication: { branch: string; commit: string },
) {
  if (
    !/^[a-f0-9]{7,40}$/.test(publication.commit) ||
    !publication.branch.trim()
  )
    throw new Error('Publication branch and commit are required')
  if (resolve(source) === resolve(destination))
    throw new Error('Source must differ from destination')
  const files = Object.fromEntries(
    await Promise.all(
      contractFiles.map(async (file) => [
        file,
        await readFile(resolve(source, file)),
      ]),
    ),
  )
  const data = await validateDetectionFiles(files)
  // Validate the entire bundle before touching the canonical snapshot. Never scan
  // the source directory: images, weights, credentials and extras are not read.
  for (const file of contractFiles) {
    const target = resolve(destination, file)
    await mkdir(dirname(target), { recursive: true })
    await writeFile(target, files[file])
  }
  const provenance = {
    repository: 'https://github.com/Kazzu00/flir-leakage-pipeline',
    branch: publication.branch,
    publication_commit: publication.commit,
    source_commit: data.manifest.source_commit,
  }
  await writeFile(
    resolve(destination, '../snapshot-provenance.json'),
    JSON.stringify(provenance, null, 2) + '\n',
  )
  return data.manifest
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const { values } = parseArgs({
    options: {
      repo: { type: 'string' },
      'export-dir': { type: 'string' },
      branch: { type: 'string' },
      'publication-commit': { type: 'string' },
    },
  })
  const repo = values.repo ?? process.env.FLIR_PIPELINE_REPO
  const source =
    values['export-dir'] ??
    (repo ? resolve(repo, 'exports/frontend/detection') : undefined)
  if (!source)
    throw new Error(
      'Use --repo <pipeline-repo>, FLIR_PIPELINE_REPO, or --export-dir <contract-directory> with --branch and --publication-commit',
    )
  const git = (args: string[]) =>
    repo
      ? execFileSync('git', ['-C', repo, ...args], { encoding: 'utf8' }).trim()
      : ''
  if (
    repo &&
    !values['export-dir'] &&
    git(['status', '--porcelain', '--', 'exports/frontend/detection'])
  ) {
    throw new Error(
      'Publish the clean upstream export before importing with Git HEAD provenance',
    )
  }
  const branch = values.branch ?? git(['branch', '--show-current'])
  const commit = values['publication-commit'] ?? git(['rev-parse', 'HEAD'])
  const destination = fileURLToPath(
    new URL('../src/features/detection/snapshot', import.meta.url),
  )
  const manifest = await syncDetection(source, destination, { branch, commit })
  console.log(
    `Imported ${contractFiles.length} verified contract files; ${manifest.completed_runs}/${manifest.expected_runs} runs. Publication: ${commit}; generator: ${manifest.source_commit}.`,
  )
}
