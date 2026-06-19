import os
from dotenv import load_dotenv
from openai import OpenAI

from ingest import GithubIngest
from rag_helper import RAGBase


load_dotenv()


query = "How does the agentic loop keep calling the model until it stops?"


ingest = GithubIngest(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = ingest.read_documents()
print(f"Loaded {len(documents)} documents")

index = ingest.build_index(use_chunks=True)
print(f"Chunks: {len(ingest.chunks)}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

rag = RAGBase(
    index=index,
    llm_client=client,
    model="gpt-5.4-mini",
)

answer, response, prompt, search_results = rag.rag(query)

print("\nRetrieved files:")
for doc in search_results:
    print("-", doc["filename"])

print("\nAnswer:")
print(answer)

print("\nPrompt characters:")
print(len(prompt))

print("\nUsage:")
print(response.usage)

print("\nInput tokens:")
print(response.usage.input_tokens)