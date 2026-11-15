import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import {
  payloadNames,
  contractFiles,
  type DetectionExport,
} from '@/features/detection/schema'

export const detectionDirectory = resolve('src/features/detection/snapshot')
export function detectionFiles() {
  return Object.fromEntries(
    contractFiles.map((file) => [
      file,
      new Uint8Array(readFileSync(resolve(detectionDirectory, file))),
    ]),
  )
}
export function detectionFixture(): DetectionExport {
  return Object.fromEntries(
    payloadNames.map((name) => [
      name,
      JSON.parse(
        readFileSync(resolve(detectionDirectory, `${name}.json`), 'utf8'),
      ),
    ]),
  ) as DetectionExport
}
