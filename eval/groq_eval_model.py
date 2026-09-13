from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

class GroqEvalModel(DeepEvalBaseLLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

        self.model = ChatGroq(
            model=model_name,
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return self.model_name