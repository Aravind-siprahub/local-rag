export interface ModelOption {
  id: string
  name: string
  description: string
  type: 'text' | 'vision'
  provider?: 'ollama' | 'openrouter' | 'nvidia' | 'omniroute'
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'qwen3:8b',
    name: 'qwen3:8b (Local)',
    description: 'Local Text & Document RAG',
    type: 'text',
    provider: 'ollama',
  },
  {
    id: 'qwen3-vl:4b',
    name: 'qwen3-vl:4b (Local Vision)',
    description: 'Multimodal Image Analysis',
    type: 'vision',
    provider: 'ollama',
  },
  {
    id: 'google/gemma-4-31b-it:free',
    name: 'Gemma 4 31B (OpenRouter)',
    description: 'Open-Source Cloud Model via OpenRouter',
    type: 'text',
    provider: 'openrouter',
  },
  {
    id: 'nvidia/nemotron-4-340b-instruct',
    name: 'Nemotron 4 340B (NVIDIA Build)',
    description: 'Accelerated Open Model via NVIDIA Build API',
    type: 'text',
    provider: 'nvidia',
  },
  {
    id: 'omniroute/auto',
    name: 'OmniRoute Gateway (Local Proxy)',
    description: 'Resilient Multi-Provider Routing & Quota Auto-Fallback',
    type: 'text',
    provider: 'omniroute',
  },
]

