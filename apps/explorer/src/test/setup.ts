import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'
import { createElement } from 'react'
afterEach(cleanup)
// Chart interaction is exercised in Chromium; unit tests isolate UI and contracts.
vi.mock('@/components/visualization/Chart', () => ({
  Chart: ({ label }: { label: string }) =>
    createElement('div', { role: 'img', 'aria-label': label }),
}))
