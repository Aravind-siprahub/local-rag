export interface ModelOption {
  id: string
  name: string
  description: string
  type: 'text' | 'vision'
}

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'qwen3:8b',
    name: 'qwen3:8b',
    description: 'Text and Document RAG',
    type: 'text',
  },
  {
    id: 'qwen3-vl:4b',
    name: 'qwen3-vl:4b',
    description: 'Multimodal Image Analysis',
    type: 'vision',
  },
]
