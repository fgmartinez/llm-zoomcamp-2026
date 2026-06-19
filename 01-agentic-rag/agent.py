import os
from dotenv import load_dotenv
from openai import OpenAI

from agents import Agent, function_tool, Runner

from ingest import GithubIngest
from rag_helper import RAGBase

load_dotenv()

# This counter answers the homework question.
# Every time the agent calls the search tool, we increment it.
calls_to_search = 0


def build_agentic_rag() -> RAGBase:
    """
    Build the retrieval layer used by the agent.

    This reuses the code we already built:
    - GithubIngest reads lesson files from GitHub.
    - GithubIngest chunks long lesson pages.
    - GithubIngest builds a minsearch index over those chunks.
    - RAGBase wraps the index and gives us search/build_context helpers.

    Important:
    This function does NOT call the LLM.
    It only prepares the searchable knowledge base.
    """

    ingest = GithubIngest(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    documents = ingest.read_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = ingest.chunk_documents(size=2000, step=1000)
    print(f"Chunks: {len(chunks)}")
    index = ingest.build_index(use_chunks=True)

    return RAGBase(
        index=index,
        llm_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        model="gpt-5.4-mini")

# Build the retrieval layer once.
# The tool below will reuse this object every time the agent decides to search.
course_rag = build_agentic_rag()


@function_tool
def search_tool(query: str) -> str:
    """
    Search the LLM Zoomcamp lesson chunks for relevant information.

    Use this tool when you need course-specific information about:
    - plain RAG
    - agentic RAG
    - agentic loop
    - tool calling
    - function calling
    - how agents decide whether to continue or stop

    Args:
        query: Search query with keywords related to the student's question.

    Returns:
        Formatted search results with filename, chunk offset, and content.
    """
     
    global calls_to_search
    calls_to_search += 1

    print(f"Search tool called {calls_to_search} times")

    search_results = course_rag.search(query, num_results=5)
    context = course_rag.build_context(search_results)

    return context


agent = Agent(
    name="Course Teaching Assistant",
    model="gpt-5.4-mini",
    instructions="""
You're a course teaching assistant. Answer the student's question using the search tool.
Make multiple searches with different keywords before answering.

When answering, explain clearly:
1. how the agentic loop works
2. how it decides whether to continue
3. how it differs from plain RAG

Base your answer only on the search results.
""",
    tools=[search_tool],
)


if __name__ == "__main__":
    question = "How does the agentic loop work, and how is it different from plain RAG?"

    result = Runner.run_sync(agent, question)

    print("\nFinal answer:")
    print(result.final_output)

    print("\nSearch tool calls:")
    print(calls_to_search)

    

