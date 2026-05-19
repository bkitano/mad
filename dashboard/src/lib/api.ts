import { supabase } from './supabase'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export { API_URL }

/**
 * Wrapper around fetch that automatically attaches the Supabase access token
 * as a Bearer token in the Authorization header.
 */
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = path.startsWith('http') ? path : `${API_URL}${path}`
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token

  const headers = new Headers(init?.headers)
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  return fetch(url, { ...init, headers })
}
