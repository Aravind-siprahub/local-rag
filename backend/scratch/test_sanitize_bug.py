import sys
from app.llm.sanitize import sanitize_response

text = """We are given a user query about SipraOne.
Let's extract information from the chunks.
Chunk 1 says SipraOne is deployed on Azure VM.
We have to be careful.
Which one to use?
Wait...
SipraOne was deployed on an Azure Ubuntu VM."""

print(repr(sanitize_response(text)))
