export function Pager({
  page,
  count,
  size,
  setPage,
}: {
  page: number
  count: number
  size: number
  setPage: (page: number) => void
}) {
  return (
    <div className="toolbar">
      <button disabled={page === 0} onClick={() => setPage(page - 1)}>
        Previous page
      </button>
      <span>
        {count ? page * size + 1 : 0}–{Math.min(count, (page + 1) * size)} /{' '}
        {count}
      </span>
      <button
        disabled={(page + 1) * size >= count}
        onClick={() => setPage(page + 1)}
      >
        Next page
      </button>
    </div>
  )
}
