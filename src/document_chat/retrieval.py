from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from typing import Optional,List
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import CustomException
from utils.model_utils import ModelLoader
from prompt.prompt_analyzer import PROMPT_REGISTRY
from model.base_model import PromptType
import sys
from operator import itemgetter
from dotenv import load_dotenv
from typing import Dict, Any
import os

load_dotenv()


class ConversationalRAG:
    def __init__(self, retriever=None, session_id: Optional[str]=None, retriver=None):
        """
        Initializes the DocumentRetriever with the path for the FAISS index.
        :param faiss_index_path: Directory where FAISS index is stored.
        :param session_id: Unique identifier for the session, defaults to current timestamp.
        """
        try:

            self.session_id = session_id or "document_chat_session"
            
            #self.parser = StrOutputParser()
            
            self.model = ModelLoader()
            self.llm = self.model.load_llm()
            self.embeddings = self.model.load_embeddings()
            self.qa_prompt = PROMPT_REGISTRY.get(PromptType.CONTEXT_QA.value)
            self.rewriter_prompt = PROMPT_REGISTRY.get(PromptType.CONTEXTUALIZE_QUESTION.value)
            
            self.retriever = retriever or retriver

            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()
            log.info("Document Retriever successfully initialized")
            
        except Exception as e:
            log.error("Error in initialization DocumentRetriever")
            raise CustomException(str(e), sys)
        
    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load FAISS vectorstore from disk and build retriever + LCEL chain.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # ok if you trust the index
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise CustomException("Loading error in ConversationalRAG", sys)

    def Invoke(self,user_query:str,chat_history:Optional[List[BaseMessage]]=None):
        try:
            if self.main_chain is None:
                raise CustomException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys)
            self.chat_history = chat_history or []
            self.payload = {"user_input":user_query, "chat_history":self.chat_history}

            if self.retriever is None:
                log.error("Retriever is not initialized")
                raise CustomException("Retriever is not initialized", sys)

            response = self.main_chain.invoke(self.payload)
            log.info("Successfully generated answer from DocumentRetriever")
            return response
        except Exception as e:
            log.error("Error in invoking DocumentRetriever")
            raise CustomException(f"Error generating answer in invoke: {e}", sys)

    def invoke(self, user_query: str, chat_history: Optional[List[BaseMessage]] = None):
        return self.Invoke(user_query, chat_history)

     
    def _create_retrivel(self,documents):
        """
        Builds a LangChain retriever using the provided documents.
        :param documents: List of Document objects to be used for retrieval.
        :return: A LangChain retriever object.
        """
        try:
            if not documents:
                raise CustomException("No documents provided for retrieval", sys)
            log.info("Building LangChain retriever")
            vector_store = FAISS.from_documents(documents, self.embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            log.info("LangChain retriever successfully built")
            return retriever
        except Exception as e:
            log.error(f"Error building LangChain retriever: {e}")
            raise CustomException(f"Error building LangChain retriever: {e}", sys)
        


    def _log_rewritten(self, question: str) -> str:
        log.info(f"Rewritten question: {question}")
        return question

    def _log_docs(self, docs):
        log.info(f"Retrieved {len(docs)} documents")
        return docs

    def _build_lcel_chain(self):
        try:
            # 1) Rewrite user question with chat history context
            if self.retriever is None:
                raise CustomException("No retriever set before building chain", sys)
            self.question_rewritter = (
                {"user_input": itemgetter("user_input"), "chat_history": itemgetter("chat_history")}
                | self.rewriter_prompt
                | self.llm
                | StrOutputParser()
                | self._log_rewritten
            )
            
            retrieve_docs = self.question_rewritter | self.retriever | self._log_docs | self._format_doc
            
            # 2) Main chain that combines the rewritten question, retrieved documents, and chat history
            self.main_chain = (
                {"context": retrieve_docs, "user_input": itemgetter("user_input"), "chat_history": itemgetter("chat_history")}
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            log.info("LCEL chain successfully built")
        except Exception as e:
            log.error(f"Error building LCEL chain: {e}")
            raise CustomException("Error building LCEL chain", sys)


    def _format_doc(self,docs):
        """
        Formats the retrieved documents into a string for further processing.
        :param docs: List of Document objects retrieved from the retriever.
        :return: A formatted string containing the content of the documents.
        """
        try:
            if not docs:
                raise CustomException("No documents to format", sys)
            formatted_docs = "\n\n".join([doc.page_content for doc in docs])
            log.info("Documents successfully formatted")
            return formatted_docs
        except Exception as e:
            log.error(f"Error formatting documents: {e}")
            raise CustomException(f"Error formatting documents: {e}", sys)
        



