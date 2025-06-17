from pathlib import Path
from typing import AsyncIterator, Iterator, List, Union, Optional, Dict

from agno.utils.log import log_info
from pydantic import Field

from agno.document import Document
from agno.document.reader.pdf_reader import PDFImageReader, PDFReader
from agno.knowledge.agent import AgentKnowledge


class PDFKnowledgeBase(AgentKnowledge):
    path: Optional[Union[str, Path]] = None

    exclude_files: List[str] = Field(default_factory=list)

    reader: Union[PDFReader, PDFImageReader] = PDFReader()

    @property
    def document_lists(self) -> Iterator[List[Document]]:
        """Iterate over PDFs and yield lists of documents.
        Each object yielded by the iterator is a list of documents.

        Returns:
            Iterator[List[Document]]: Iterator yielding list of documents
        """
        if self.path is None:
            return

        _pdf_path: Path = Path(self.path) if isinstance(self.path, str) else self.path

        if _pdf_path.exists() and _pdf_path.is_dir():
            for _pdf in _pdf_path.glob("**/*.pdf"):
                if _pdf.name in self.exclude_files:
                    continue
                yield self.reader.read(pdf=_pdf)
        elif _pdf_path.exists() and _pdf_path.is_file() and _pdf_path.suffix == ".pdf":
            if _pdf_path.name in self.exclude_files:
                return
            yield self.reader.read(pdf=_pdf_path)

        else:
            raise FileNotFoundError(f"PDFKnowledgeBase: path `{self.path}` is not a valid PDF file or directory.")

    @property
    async def async_document_lists(self) -> AsyncIterator[List[Document]]:
        """Iterate over PDFs and yield lists of documents.
        Each object yielded by the iterator is a list of documents.

        Returns:
            Iterator[List[Document]]: Iterator yielding list of documents
        """
        if self.path is None:
            return

        _pdf_path: Path = Path(self.path) if isinstance(self.path, str) else self.path

        if _pdf_path.exists() and _pdf_path.is_dir():
            for _pdf in _pdf_path.glob("**/*.pdf"):
                if _pdf.name in self.exclude_files:
                    continue
                yield await self.reader.async_read(pdf=_pdf)
        elif _pdf_path.exists() and _pdf_path.is_file() and _pdf_path.suffix == ".pdf":
            if _pdf_path.name in self.exclude_files:
                return
            yield await self.reader.async_read(pdf=_pdf_path)

        else:
            raise FileNotFoundError(f"PDFKnowledgeBase: `path` is not a valid PDF file or directory.")

    def load(
            self,
            recreate: bool = False,
            upsert: bool = True,
            skip_existing: bool = True,
            filters: Optional[Dict[str, Union[str, int]]] = None,
    ) -> None:
        """安全加载 PDF 到向量库（支持空 path，自动跳过）"""
        if self.path is None:
            log_info("PDFKnowledgeBase.load() Skipped: path not set, no document loaded.")
            return

        super().load(
            recreate=recreate,
            upsert=upsert,
            skip_existing=skip_existing,
        )
