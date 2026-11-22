import { useState, type ReactNode } from 'react'
import { Dialog } from 'radix-ui'
import { X, SlidersHorizontal } from 'lucide-react'
/** Mount expensive audit tables only on request; restore trigger focus on close. */
export function TechnicalDetailsDrawer({
  title = 'Technical details',
  children,
}: {
  title?: string
  children: ReactNode
}) {
  const [open, setOpen] = useState(false)
  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger className="details-trigger">
        <SlidersHorizontal size={15} />
        {title}
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content className="technical-drawer">
          <div className="drawer-heading">
            <Dialog.Title>{title}</Dialog.Title>
            <Dialog.Close aria-label="Close technical details">
              <X size={20} />
            </Dialog.Close>
          </div>
          <Dialog.Description>
            Source identities, reported evidence and export integrity. Export
            validation is not scientific verification.
          </Dialog.Description>
          {open && children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
