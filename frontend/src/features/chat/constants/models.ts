export interface ModelOption {
  id: string
  name: string
  description: string
  type: 'text' | 'vision'
  provider?: 'ollama' | 'openrouter' | 'nvidia' | 'omniroute'
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'local-rag',
    name: 'OmniRoute Gateway (local-rag)',
    description: 'NVIDIA NIM + OpenRouter Fallback (1-2s response)',
    type: 'text',
    provider: 'omniroute',
  },
  {
    id: 'local-rag-vision',
    name: 'OmniRoute Vision (local-rag-vision)',
    description: 'Nemotron Omni Multimodal Gateway (1-2s response)',
    type: 'vision',
    provider: 'omniroute',
  },
  {
    id: 'nvidia/nemotron-3.5-lightning-30b-a3b',
    name: 'Nemotron 3.5 Lightning (NVIDIA Cloud)',
    description: 'High-speed NVIDIA NIM Cloud Reasoning (1-2s response)',
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
    id: 'nvidia/meta/llama-3.2-11b-vision-instruct',
    name: 'Llama 3.2 11B Vision (NVIDIA Cloud)',
    description: 'High-speed NVIDIA NIM Multimodal Vision (<1s response)',
    type: 'vision',
    provider: 'nvidia',
  },
  {
    id: 'qwen3-vl:4b',
    name: 'qwen3-vl:4b (Local Vision)',
    description: 'Multimodal Image Analysis',
    type: 'vision',
    provider: 'ollama',
  },
]

