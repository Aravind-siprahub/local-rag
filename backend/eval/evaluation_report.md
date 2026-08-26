# Local RAG Evaluation Report

## Retrieval
Hit@1: 10.0%
Hit@3: 10.0%
Hit@5: 10.0%
Hit@10: 10.0%

## Citation
Citation accuracy: 10.0%
Correct source rate: 10.0%

## Answer
Grounded answer rate: 90.0%
Expected facts rate: 1.8%
Unsupported answer rate: 10.0%

## No-answer
Correct refusal rate: 0.0%

## Performance
Average retrieval latency: 2793.3 ms
Average total latency: 2793.3 ms

## PER-QUESTION RESULTS

[1] Q: What is Talk to My Data?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2599 ms (Retrieval: 2599 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[2] Q: What is Vanna?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2558 ms (Retrieval: 2558 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[3] Q: What is the AI Router?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2760 ms (Retrieval: 2760 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[4] Q: What is pgvector?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3048 ms (Retrieval: 3048 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[5] Q: What is DuckDB in this context?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2730 ms (Retrieval: 2730 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[6] Q: What is recursive text chunking?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2894 ms (Retrieval: 2894 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[7] Q: What is nomic-embed-text?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2651 ms (Retrieval: 2651 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[8] Q: What is an embedding vector?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2632 ms (Retrieval: 2632 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[9] Q: What is cosine similarity distance?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2890 ms (Retrieval: 2890 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[10] Q: What is DDL in database context?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2615 ms (Retrieval: 2615 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[11] Q: How does RAG differ from Text-to-SQL?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2912 ms (Retrieval: 2912 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[12] Q: Compare structured data processing with unstructured document processing.
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2698 ms (Retrieval: 2698 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[13] Q: What is the difference between top_k candidates and final_context passages?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2932 ms (Retrieval: 2932 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[14] Q: Compare read-only database roles with admin roles in Text-to-SQL.
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2617 ms (Retrieval: 2617 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[15] Q: What is the difference between dense vector search and full-text keyword search?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2928 ms (Retrieval: 2928 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[16] Q: Compare local disk storage with Supabase cloud storage.
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2636 ms (Retrieval: 2636 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[17] Q: How does DuckDB differ from PostgreSQL in this system?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2686 ms (Retrieval: 2686 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[18] Q: Compare system settings stored in database vs environment variables.
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2623 ms (Retrieval: 2623 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[19] Q: What is the difference between SSE streaming and standard HTTP responses?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2899 ms (Retrieval: 2899 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[20] Q: Compare chunk size vs chunk overlap.
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2714 ms (Retrieval: 2714 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[21] Q: Why must RAG and Text-to-SQL remain separate technical mechanisms?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2921 ms (Retrieval: 2921 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[22] Q: Why does Text-to-SQL require a read-only database role?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2648 ms (Retrieval: 2648 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[23] Q: Why is an AI Router necessary in front of a single chat box?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3203 ms (Retrieval: 3203 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[24] Q: Why is Reciprocal Rank Fusion (RRF) used in reranking?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3252 ms (Retrieval: 3252 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[25] Q: Why are thinking tags sanitized from output responses?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2764 ms (Retrieval: 2764 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[26] Q: Why is chunk overlap important during document parsing?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2662 ms (Retrieval: 2662 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[27] Q: Why does the system enforce strict non-hallucination rules?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3209 ms (Retrieval: 3209 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[28] Q: Why is Vanna trained on schema DDL statements?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2742 ms (Retrieval: 2742 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[29] Q: Why are citations attached to generated RAG answers?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2598 ms (Retrieval: 2598 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[30] Q: Why does LLM timeout need to be configured for local execution?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2723 ms (Retrieval: 2723 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[31] Q: How are document passages retrieved and passed to the LLM?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2674 ms (Retrieval: 2674 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[32] Q: How does Vanna translate natural language questions into SQL?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2634 ms (Retrieval: 2634 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[33] Q: How does DuckDB execute SQL queries locally?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2550 ms (Retrieval: 2550 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[34] Q: How is conversation history summarized when context limits are reached?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2731 ms (Retrieval: 2731 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[35] Q: How does the output sanitizer handle unclosed thinking tags?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2647 ms (Retrieval: 2647 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[36] Q: How does pgvector compute vector similarity?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2715 ms (Retrieval: 2715 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[37] Q: How does the application manage database connection sessions?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2884 ms (Retrieval: 2884 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[38] Q: How are system settings updated dynamically?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2591 ms (Retrieval: 2591 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[39] Q: How are uploaded files parsed into text chunks?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2837 ms (Retrieval: 2837 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[40] Q: How does the frontend handle dark mode themes?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2560 ms (Retrieval: 2560 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[41] Q: What is the overall architecture of Talk to My Data?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2955 ms (Retrieval: 2955 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[42] Q: What database tables exist in the PostgreSQL schema?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2575 ms (Retrieval: 2575 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[43] Q: Where does vector retrieval fit in the query lifecycle?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3049 ms (Retrieval: 3049 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[44] Q: What is the role of FastAPI in the backend?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2691 ms (Retrieval: 2691 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[45] Q: How does the backend interface with Ollama?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2546 ms (Retrieval: 2546 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[46] Q: What security boundaries protect database operations?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2577 ms (Retrieval: 2577 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[47] Q: What role does React playing in the frontend?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3035 ms (Retrieval: 3035 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[48] Q: What role does Tailwind CSS perform in the application?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2604 ms (Retrieval: 2604 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[49] Q: How are environment configurations managed?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2860 ms (Retrieval: 2860 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[50] Q: How is health monitoring implemented across system dependencies?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2571 ms (Retrieval: 2571 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[51] Q: How does Text-to-SQL execute queries safely?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2928 ms (Retrieval: 2928 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[52] Q: Can Text-to-SQL execute arbitrary data modification statements?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2764 ms (Retrieval: 2764 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[53] Q: What training data does Vanna require for Text-to-SQL?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2881 ms (Retrieval: 2881 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[54] Q: How does DuckDB execute SQL over local CSV or Excel files?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2769 ms (Retrieval: 2769 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[55] Q: What failure modes occur during Text-to-SQL generation?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2540 ms (Retrieval: 2540 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[56] Q: What is a read-only database role?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2807 ms (Retrieval: 2807 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[57] Q: How are structured query results formatted for user presentation?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3026 ms (Retrieval: 3026 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[58] Q: Why is SQL query validation performed prior to execution?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2872 ms (Retrieval: 2872 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[59] Q: How does Text-to-SQL handle complex relational joins?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2858 ms (Retrieval: 2858 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[60] Q: What role does PostgreSQL play in structured data handling?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2871 ms (Retrieval: 2871 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[61] Q: What is RAG in this system?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2965 ms (Retrieval: 2965 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[62] Q: How does dense vector similarity search retrieve relevant chunks?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2915 ms (Retrieval: 2915 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[63] Q: What is the purpose of hybrid search in retrieval?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3021 ms (Retrieval: 3021 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[64] Q: How are prompt context passages formatted for the LLM?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2590 ms (Retrieval: 2590 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[65] Q: What non-hallucination rules are enforced during RAG answer generation?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2877 ms (Retrieval: 2877 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[66] Q: What metadata is attached to each retrieved chunk?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2973 ms (Retrieval: 2973 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[67] Q: What similarity threshold is applied during vector search?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3035 ms (Retrieval: 3035 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[68] Q: How does the system handle multi-passage evidence merging?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2612 ms (Retrieval: 2612 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[69] Q: How are citations constructed in the API response payload?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2926 ms (Retrieval: 2926 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[70] Q: What vector dimension size is produced by nomic-embed-text?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2581 ms (Retrieval: 2581 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[71] Q: Given that RAG retrieves chunks, what happens if no relevant chunks are found?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2549 ms (Retrieval: 2549 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[72] Q: If a user follows up with 'Tell me more', how is context maintained?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2680 ms (Retrieval: 2680 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[73] Q: What happens if context length exceeds MAX_CONTEXT_TOKENS?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2933 ms (Retrieval: 2933 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[74] Q: If Ollama emits thinking tags, how are they removed before returning to user?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2706 ms (Retrieval: 2706 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[75] Q: What happens if a user re-uploads an existing document?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2730 ms (Retrieval: 2730 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[76] Q: How does the system handle hybrid questions requiring both document search and SQL querying?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2619 ms (Retrieval: 2619 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[77] Q: What happens if a hybrid query returns SQL results but no matching documents?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3001 ms (Retrieval: 3001 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[78] Q: How are citations displayed when a hybrid answer includes both text and tabular data?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2784 ms (Retrieval: 2784 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[79] Q: What safety rules apply when hybrid execution runs SQL and RAG concurrently?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2570 ms (Retrieval: 2570 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[80] Q: How does the AI Router decide if a question is hybrid?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3041 ms (Retrieval: 3041 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[81] Q: What is the stock price of Apple Inc today?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2766 ms (Retrieval: 2766 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[82] Q: Who won the 2022 FIFA World Cup?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2869 ms (Retrieval: 2869 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[83] Q: How do you configure Kubernetes ingress controllers?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2854 ms (Retrieval: 2854 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[84] Q: What is the recipe for baking sourdough bread?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2972 ms (Retrieval: 2972 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[85] Q: Who was the first President of the United States?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2687 ms (Retrieval: 2687 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[86] Q: What is the capital city of Australia?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2746 ms (Retrieval: 2746 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[87] Q: What is Einstein's theory of general relativity?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2776 ms (Retrieval: 2776 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[88] Q: How do you tie a windsor knot tie?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2566 ms (Retrieval: 2566 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[89] Q: What is the speed of light in vacuum?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2934 ms (Retrieval: 2934 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[90] Q: Who directed the movie Inception?
    Status: FAIL
    Expected Source: []
    Retrieved Source: []
    Hit@K: Hit@1: True, Hit@3: True, Hit@5: True
    Citation Result: PASS
    Answer Result: FAIL
    Latency: 2740 ms (Retrieval: 2740 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Failed to refuse out-of-context question

[91] Q: How does it work?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3107 ms (Retrieval: 3107 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[92] Q: Tell me about security.
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2640 ms (Retrieval: 2640 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[93] Q: What are the limitations?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2893 ms (Retrieval: 2893 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[94] Q: Is it production ready?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3281 ms (Retrieval: 3281 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[95] Q: What data can I ask about?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2966 ms (Retrieval: 2966 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[96] Q: What models are supported?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2552 ms (Retrieval: 2552 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[97] Q: How are errors handled?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2985 ms (Retrieval: 2985 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[98] Q: What vector database is used?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2816 ms (Retrieval: 2816 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[99] Q: How does context chunking work?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 3005 ms (Retrieval: 3005 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

[100] Q: What is the primary architectural goal of Talk to My Data?
    Status: FAIL
    Expected Source: ['PRD_Talk_to_My_Data.docx']
    Retrieved Source: []
    Hit@K: Hit@1: False, Hit@3: False, Hit@5: False
    Citation Result: FAIL
    Answer Result: FAIL
    Latency: 2723 ms (Retrieval: 2723 ms)
    Issues: HTTP 404: {"detail":"ChatSession with id=UUID('00000000-0000-0000-0000-000000000000') was not found."}, Expected document missing from top-5 retrieved context, Citation validation failed or missing expected source, Answer correctness/groundedness threshold not met

