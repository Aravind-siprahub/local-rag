import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.rag.intent_router import classify, Route
import uuid

q1 = "what frontend and backend are using talk to my data"
q2 = "pls give info on how nginx is uploaded"

print(f"Q1: {q1} -> {classify(q1, document_titles=['PRD_Talk_to_My_Data.docx'])}")
print(f"Q2: {q2} -> {classify(q2)}")
