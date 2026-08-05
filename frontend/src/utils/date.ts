import { format, formatDistanceToNow, isValid, parseISO } from 'date-fns'

export function parseDate(value: string | null | undefined): Date | null {
  if (!value) {
    return null
  }

  const parsed = parseISO(value)
  return isValid(parsed) ? parsed : null
}

export function formatRelativeTime(value: string | null | undefined): string {
  const date = parseDate(value)
  if (!date) {
    return '—'
  }

  return formatDistanceToNow(date, { addSuffix: true })
}

export function formatDateTime(value: string | null | undefined): string {
  const date = parseDate(value)
  if (!date) {
    return '—'
  }

  return format(date, 'MMM d, yyyy · h:mm a')
}

export function sortByCreatedAtDesc<T extends { created_at: string }>(items: T[]): T[] {
  return [...items].sort(
    (left, right) => new Date(right.created_at).getTime() - new Date(left.created_at).getTime(),
  )
}
