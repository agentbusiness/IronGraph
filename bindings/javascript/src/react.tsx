import { createContext, createElement, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Client, type QueryRequest, type QueryResult } from './index.js'

const IronGraphContext = createContext<Client | null>(null)

export function IronGraphProvider({ baseUrl, children }: { baseUrl: string | URL; children: ReactNode }) {
  const client = useMemo(() => new Client(baseUrl), [String(baseUrl)])
  return createElement(IronGraphContext.Provider, { value: client }, children)
}

export function useIronGraphClient(): Client {
  const client = useContext(IronGraphContext)
  if (!client) throw new Error('useIronGraphClient must be used within IronGraphProvider')
  return client
}

export function useIronGraphQuery(request: Omit<QueryRequest, 'signal'> | null) {
  const client = useIronGraphClient()
  const [result, setResult] = useState<QueryResult>()
  const [error, setError] = useState<unknown>()
  const [loading, setLoading] = useState(request !== null)
  const signature = JSON.stringify(request)

  useEffect(() => {
    if (!request) {
      setLoading(false)
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError(undefined)
    client.query({ ...request, signal: controller.signal })
      .then(setResult)
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setError(caught)
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [client, signature])

  return { result, error, loading }
}
