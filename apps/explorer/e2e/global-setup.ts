import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join, resolve } from 'node:path'
import { execFileSync } from 'node:child_process'

export default function prepareSyntheticEvidence() {
  const backend = resolve(process.env.FLIR_AUDIT_BACKEND ?? '../..')
  const python =
    process.env.FLIR_AUDIT_PYTHON ??
    join(
      backend,
      '.venv',
      process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
    )
  const output = mkdtempSync(join(tmpdir(), 'flir-e2e-SYNTHETIC-'))
  execFileSync(
    python,
    ['-B', join(backend, 'tests/prepare_explorer_e2e.py'), output],
    {
      cwd: backend,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONPATH: [join(backend, 'src'), process.env.PYTHONPATH]
          .filter(Boolean)
          .join(delimiter),
        PYTHONDONTWRITEBYTECODE: '1',
        HF_HUB_OFFLINE: '1',
        TRANSFORMERS_OFFLINE: '1',
      },
    },
  )
  process.env.FLIR_E2E_SYNTHETIC_BUNDLE = output
}
