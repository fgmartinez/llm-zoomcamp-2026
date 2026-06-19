from typing import Callable
from gitsource import GithubRepositoryDataReader
from minsearch import Index
from gitsource import chunk_documents


class GithubIngest:
    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        commit_id: str,
        allowed_extensions: set[str] = {"md"},
        filename_filter: Callable[[str], bool] | None = None,
    ) -> None:
        self.reader = GithubRepositoryDataReader(
            repo_owner=repo_owner,
            repo_name=repo_name,
            commit_id=commit_id,
            allowed_extensions=allowed_extensions,
            filename_filter=filename_filter,
        )
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self.index = None

    def read_files(self):
        return self.reader.read()

    def read_documents(self):
        self.documents = [file.parse() for file in self.read_files()]
        return self.documents
    
    def chunk_documents(self, size: int = 2000, step: int = 1000):
        if not self.documents:
            self.read_documents()

        self.chunks = chunk_documents(
            self.documents,
            size=size,
            step=step,
        )

        return self.chunks

    def build_index(self, use_chunks: bool = False):
        if use_chunks:
            docs_to_index = self.chunk_documents(size=2000, step=1000)
        else:
            if not self.documents:
                self.read_documents()
            docs_to_index = self.documents

        self.index = Index(
            text_fields=["content"],
            keyword_fields=["filename"],
        )

        self.index.fit(docs_to_index)
        print(f"Documents: {len(self.documents)}")
        print(f"Chunks: {len(self.chunks)}")

        return self.index