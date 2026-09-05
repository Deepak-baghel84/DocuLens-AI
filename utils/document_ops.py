from pathlib import Path
import os
from typing import Iterable, List
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import CustomException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

def load_documents(file_paths:Iterable[Path])-> List[Document]:
    docs: List[Document] = []
    try:
        #if not file_paths or len(file_paths) == 0:
         #   log.error("No file paths provided for loading documents.")
          #  raise CustomException("No file paths provided for loading documents.", sys)

        log.info(f"Loading {len(file_paths)} documents")
        for file_path in file_paths :

            file_name = os.path.basename(file_path)
            ext= Path(file_path).suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(file_path))
            elif ext == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(file_path))
                continue
                
        
            doc = loader.load()
            docs.extend(f"Document: {file_name}\n{d.page_content}" for d in doc)
        log.info(f"Loaded {len(docs)} pages successfully")
        docs = [Document(page_content=d) if isinstance(d, str) else d for d in docs]
        return docs
    except Exception:
        log.error("Error loading document")
        raise CustomException("Error during load document")
    
def concat_for_analysis(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"

#---------- Helper Classes and Functions ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()


def read_pdf_via_handler(handler, path: str) -> str:
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")