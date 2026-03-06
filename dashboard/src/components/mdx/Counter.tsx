import { useState } from 'react'

interface CounterProps {
  initialValue?: number
}

export function Counter({ initialValue = 0 }: CounterProps) {
  const [count, setCount] = useState(initialValue)

  return (
    <div className="my-4 inline-flex items-center gap-3 p-3 bg-gray-100 rounded">
      <button
        onClick={() => setCount(c => c - 1)}
        className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded font-mono"
      >
        -
      </button>
      <span className="font-mono text-lg min-w-[3ch] text-center">{count}</span>
      <button
        onClick={() => setCount(c => c + 1)}
        className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded font-mono"
      >
        +
      </button>
    </div>
  )
}
