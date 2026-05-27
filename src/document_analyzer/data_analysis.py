from exception.custom_exception import CustomException
from logger import GLOBAL_LOGGER as log
from utils.model_utils import ModelLoader
from prompt.prompt_analyzer import PROMPT_REGISTRY
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from model.base_model import Metadata
import sys

class  DocumentAnalyzer:
    """Class to perform data analysis on the ingested PDF files.
    It initializes the model loader and provides methods for data analysis."""
    
    def __init__(self):
        try:
            self.load_model = ModelLoader()
            self.llm_model = self.load_model.load_llm()
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser,llm=self.llm_model)
            self._prompt = PROMPT_REGISTRY["document_analysis"]
           
        except Exception as e:
            log.error(f"Error initializing DataAnalysis: {e}")
            raise CustomException(e, sys)
        
    def analyze_document(self, document_text):
        """Method to analyze the document text and then return structured data."""

        log.info("Starting document analysis...")
        try:
           self.chain = self._prompt | self.llm_model | self.fixing_parser
           response = self.chain.invoke({"format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text})
           log.info("Document analysis completed successfully.")
           return response
        except Exception as e:
            log.error(f"Error during document analysis: {e}")
            raise CustomException(e, sys)