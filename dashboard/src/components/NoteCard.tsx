import { Link } from 'react-router-dom'
import type { NoteMeta } from '../data/note-loader'

interface NoteCardProps {
  note: NoteMeta
}

export default function NoteCard({ note }: NoteCardProps) {
  return (
    <article className="py-4">
      <Link to={note.routePath} className="group block">
        <h2 className="text-lg font-medium text-gray-900 group-hover:text-blue-600">
          {note.title}
        </h2>
        <div className="mt-1 flex items-center gap-2 text-sm text-gray-500">
          <span className="font-mono text-xs bg-gray-100 px-2 py-0.5 rounded">
            {note.project}
          </span>
          {note.date && (
            <time dateTime={note.date}>
              {new Date(note.date).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
              })}
            </time>
          )}
        </div>
        {note.description && (
          <p className="mt-2 text-gray-600 text-sm">{note.description}</p>
        )}
      </Link>
    </article>
  )
}
