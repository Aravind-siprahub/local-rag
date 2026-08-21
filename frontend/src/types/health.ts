export interface HealthResponse {
  status: 'ok' | 'error'
  database: string
  pgvector?: string
  ollama?: string
  models?: string[]
}
