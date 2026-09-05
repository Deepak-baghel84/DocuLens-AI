from logger import GLOBAL_LOGGER as log
from exception.custom_exception import CustomException 
from pathlib import Path 
import sys 
import json
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.model_utils import ModelLoader
from utils.document_ops import load_documents
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
from typing import Any, List, Optional, Iterable, Dict
from pypdf import PdfReader
from utils.file_io import save_uploaded_files,generate_session_id

class FaissManager:
    '''Class to manage FAISS index operations.
       '''
    def __init__(self,index_dir:Path,model_loader:Optional[ModelLoader]=None):
        log.info(f"Initializing FaissManager with index directory: {index_dir}")
        #print(f"Initializing FaissManager with index directory--")
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_loader = model_loader or ModelLoader()
        self.embed = self.model_loader.load_embeddings()
        
        self.meta_path = self.index_dir / "metadata.json"
        self._meta:Dict[str,Any] = {"rows":{}}
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists
        self.vs: Optional[FAISS] = None

        log.info("FaissManager initialized", index_dir=str(self.index_dir), meta_loaded=bool(self._meta["rows"]))
        
    def _exists(self)-> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta), encoding="utf-8")     #ensure_ascii=False

    def add_documents(self,docs: List[Document]):
        '''Add documents to the FAISS index, ensuring no duplicates based on content and metadata.'''
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")
        
        new_docs: List[Document] = []
        
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)
            
        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        '''Loads an existing FAISS index or creates a new one if it doesn't exist. '''
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.embed,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        
        if not texts:
            raise CustomException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.embed, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        log.info("Created new FAISS index", index=str(self.index_dir))
        return self.vs
class ChatIngestor():
    """Class to handle document ingestion and FAISS index creation for chat applications.
    """
    
    def __init__(self,temp_base: str = "Data",faiss_base: str = "faiss_index",use_session_dirs: bool = True,session_id: Optional[str] = None,):
        """
        Initializes the DocumentIngestor with paths for file storage and FAISS index.
        :param file_path: Directory where documents will be stored.
        :param session_id: Unique identifier for the session, defaults to current timestamp.
        :param faiss_index_path: Directory where FAISS index will be stored.
        """
        
        try:
            self.model_loader = ModelLoader()
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base) 
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise Exception("Initialization error in ChatIngestor", e) from e
        
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"   


        #  _remove_pdf_files(base_dir=self.temp_base)
          
    def _split(self,docs: List,*,chunk_size: int = 1000,chunk_overlap: int = 200)-> List:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(docs)     #split the document into chunks not the text 
            log.info(f"Documents split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
            return chunks
        except Exception as e:
            log.error(f"Error splitting documents: {e}")
            raise CustomException(f"Error splitting documents: {e}", sys)
        
    def create_retrivel(self,documents: Iterable,*,chunk_size: int = 1000,chunk_overlap: int = 200,k: int = 5):
        try:
            paths = save_uploaded_files(documents, self.temp_dir)     #document versioning
            docs = load_documents(paths)            #return the text from the document
            if not docs or docs == []:
                log.info("No valid text extracted from documents")
                raise ValueError("No valid text extracted")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(docs)     #split the document into chunks not the text 
            log.info(f"Documents split into {len(chunks)} chunks ")

             # create or load faiss index 
            fm = FaissManager(self.faiss_dir, self.model_loader)
            text=[c.page_content for c in chunks]
            md=[m.metadata for m in chunks]
        
            try:
                vs = fm.load_or_create(texts=text, metadatas=md)
            except Exception:
                vs = fm.load_or_create(texts=text, metadatas=md)

            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            log.error("Failed to create retriever", error=str(e))
            raise CustomException("Failed to create retriever", e) from e     
            

    
class DocHandler():
    def __init__(self,dir_path:Optional[str]="Data//analyzer_archive",session_id:Optional[str]=None):
        """Initialize the DataIngestion class with file path and session ID.
        :param file_path: Path to the PDF file to be processed.
        :param session_id: Unique identifier for the session.  
        """
        try:
          log.info(f"Initializing DataIngestion with file path: {dir_path} and session ID: {session_id}")
          self.dir_path = dir_path or os.getenv("DEFAULT_FILE_PATH",os.path.join(os.getcwd(), "data","archive_pdfs"))
          self.session_id = session_id or generate_session_id()

          self.new_dir_path = os.path.join(self.dir_path,self.session_id)
          self.new_file = os.makedirs(self.new_dir_path, exist_ok=True)
          log.info(f"Directory created at: {self.new_dir_path}")
        except:
            log.error(f"Error initializing DataIngestion with file path: {dir_path} and session ID: {session_id}")
            raise CustomException("Failed to initialize DataIngestion", sys)
                                  
    def save_pdf(self,uploaded_files):
        try:
            file_name = os.path.basename(uploaded_files.name)
            save_path = os.path.join(self.new_dir_path,file_name)
            with open(save_path, 'wb') as file:
                file.write(uploaded_files.getbuffer())

            log.info(f"PDF saved successfully at: {save_path}")
            _remove_pdf_files(base_dir=self.dir_path)
            return save_path
        except FileNotFoundError as e:
            log.error(f"File not found: {e}")
            raise CustomException(f"File not found: {e}", sys)

    def read_pdf(self,file_path:Path):
        
        try:
            text_chunks = []
            number = 1
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_chunks.append(f"file-Page ---{number}---: \n{text}")
                    number += 1  # type: ignore
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=file_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=file_path, session_id=self.session_id)
            raise Exception(f"Could not process PDF: {file_path}", e) from e


class DocumentComparator():
    def __init__(self, dir_path: str = "Data//comparator_archive", session_id: str = None):
        """
        Initialize the DataIngestion class with file path and session ID.
        :param file_path: Path to the PDF file to be processed.
        :param session_id: Unique identifier for the session.  
        """
        try:
            # Set base directory for saving PDFs
            self.base_dir_path = dir_path or os.getenv("DEFAULT_FILE_PATH", os.path.join(os.getcwd(), "data", "archive_pdfs"))
            
            self.sessionn_file = session_id or generate_session_id()
            self.session_path = Path(self.base_dir_path) / Path(self.sessionn_file)
            self.session_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error(f"Error initializing Ingestor: {e}")
            raise CustomException(e, sys)
    def save_pdf_files(self,ref_file,act_file):
        """Save the PDF files to the specified directory.
        :param ref_file: Path to the reference PDF file.
        :param act_file: Path to the actual PDF file.
        :return: Paths where the PDF files are saved.
        """
        try:
            #if not self.ref_file.name.lower().endswith('.pdf') or not self.act_file.name.lower().endswith('.pdf'):
             #   raise ValueError("One or both files are not PDFs.")
            ref_save_path = self.session_path / ref_file.name
            act_save_path = self.session_path / act_file.name
            for fobj, out in ((ref_file,ref_save_path), (act_file, act_save_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            log.info(f"PDF files saved successfully at: {self.session_path}")

            #_remove_pdf_files(base_dir=self.base_dir_path)

            return ref_save_path, act_save_path

        except FileNotFoundError as e:
            log.error(f"File not found: {e}")
            raise CustomException(f"File not found: {e}", sys)

    def read_pdf(self, file_path:Path):
        """Read the PDF file and return its content.
        :param file_path: Path to the PDF file to be read.
        :return: Text content of the PDF file.
        """
        try:
            data_block = []
            number = 1
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        data_block.append(f"file-Page ---{number}---: \n{text}")
                    number += 1
                       
            if not data_block:
                raise ValueError("No text found in the PDF file.")
            
            log.info(f"PDF read successfully from: {file_path} with {len(data_block)} pages")
            if data_block == []:
                raise ValueError("PDF file is empty or could not be read.")
            return "\n".join(data_block)
        except FileNotFoundError as e:
            log.error(f"File not found: {e}")
            raise CustomException(f"File not found: {e}", sys)
        

    def combine_pdf_text(self):
        """Combine the text from the reference and actual PDF files.
        :param ref_text: Text content of the reference PDF file.
        :param act_text: Text content of the actual PDF file.
        :return: Combined text content.
        """
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined successfully")
            return combined_text
        except Exception as e:
            log.error("Error combining documents")
            raise Exception("Error combining documents", e) from e
              #---------------Optional------------------#
def _remove_pdf_files(base_dir="Data/archive",log_dir="logs",keep_latest:int=3):
        """Remove the PDF files from the session directory."""
        try:
            _base_dir = Path(base_dir)
            _log_dir = Path(log_dir)
            sessions = sorted([f for f in _base_dir.iterdir() if f.is_dir()], reverse=True)
            print(sessions)
            if len(sessions) > keep_latest:
                for session in sessions[keep_latest:]:
                    if os.path.exists(session):
                      os.remove(session)
                log.info("Old session removed successfully ")
              
            
            logs = sorted([f for f in _log_dir.iterdir() if f.is_file() and f.suffix == '.log'], reverse=True)
            if len(logs) > keep_latest:
                for log_file in logs[keep_latest:]:
                    if os.path.exists(log_file):
                      os.remove(log_file)
                log.info("Old logs removed successfully ")

            return
        except Exception as e:
            log.error(f"Error removing files or logs: {e}")
            raise CustomException(f"Error removing files or logs: {e}", sys)