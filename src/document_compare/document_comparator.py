from logger import GLOBAL_LOGGER as log
from exception.custom_exception import CustomException
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_analyzer import PROMPT_REGISTRY
from utils.model_utils import ModelLoader
from model.base_model import SummaryResponse,PromptType
import sys
from dotenv import load_dotenv
import pandas as pd


class DocumentComparatorLLM:
    def __init__(self):
        try:
          load_dotenv()
          self.loader = ModelLoader()
          self.llm = self.loader.load_llm()
          self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
          self.fixing_parser = OutputFixingParser.from_llm(
              parser=self.parser,
              llm=self.llm
          )
          #self.parser = JsonOutputParser(pydantic_object=PromptType)

          self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]

          self.chain = self.prompt | self.llm | self.fixing_parser
          log.info("DocumentCompare initialized successfully.")
        except Exception as e:
            log.error(f"Error initializing DocumentCompare: {e}")
            raise CustomException("Error in initialization of DocumentCompare file", sys)
           
    def Document_compare(self,combined_text:str)->pd.DataFrame:
        try:
            log.info("Starting document comparison process.")
            if not combined_text:
                log.error("Combined text for comparison is empty.")
                raise ValueError("Combined text for comparison is empty.")
                
            inputs = {"combined_docs":combined_text ,
                      "format_instruction":self.parser.get_format_instructions()}
            
            response = self.chain.invoke(inputs)
            log.info("Document comparision have completed successfully.")
            return self._format_response(response)
        except Exception as e:
            log.info(f"Document comparision have an issue: {e}")
            raise CustomException("Document_compare have an issue ",sys)


    def _format_response(self,response_parsed:str)->pd.DataFrame:
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            raise CustomException("Error formatting response", sys)