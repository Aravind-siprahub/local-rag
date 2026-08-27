export interface ModelOption {
  id: string
  name: string
  description: string
  type: 'text' | 'vision'
  provider?: 'ollama' | 'openrouter' | 'nvidia' | 'omniroute'
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'auto/fast',
    name: 'OmniRoute Gateway (Auto Fast Cloud)',
    description: 'Auto-Fallback Cloud Gateway (1-3s response)',
    type: 'text',
    provider: 'omniroute',
  },
  {
    id: 'nvidia/nemotron-4-340b-instruct',
    name: 'Nemotron 4 340B (NVIDIA Cloud)',
    description: 'Accelerated Cloud Model (1-2s response)',
    type: 'text',
    provider: 'nvidia',
  },
  {
    id: 'qwen3:8b',
    name: 'qwen3:8b (Local Offline)',
    description: 'Local CPU Model (~50s response)',
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
]

