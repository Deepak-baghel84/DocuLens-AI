import os
import sys
from pathlib import Path
from typing import List
import json
from dotenv import load_dotenv

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval import evaluate

from logger import GLOBAL_LOGGER as log

from src.document_ingestion.data_ingestion import ChatIngestor
from src.document_chat.retrieval import ConversationalRAG


        # Directory containing documents to ingest and index
DEEPEVAL_INPUT_DIR = os.getenv("DEEPEVAL_INPUT_DIR","data_deep_eval")

             # Document versioning path
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "Data/multidoc_archive")    

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")   # faiss index path
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

       # Local JSON evaluation dataset
DEEPEVAL_DATASET_PATH = os.getenv("DEEPEVAL_DATASET_PATH", "eval_ques/tcs_sql_interview_evaluation_dataset-1.json")  # dataset path, questions and expected answers for evaluation


class LocalFileAdapter:
    def __init__(self, file_path: str):
        self.name = os.path.basename(file_path)
        self._file_path = file_path

    def read(self) -> bytes:
        with open(self._file_path, "rb") as f:
            return f.read()

    # For compatibility with save_uploaded_files
    def getbuffer(self) -> bytes:
        return self.read()


def list_supported_files(root: Path) -> List[Path]:
    exts = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}
    files: List[Path] = []
    for p in sorted(root.rglob("*")):    # Take the root directory and recursively search for all files, then sort them alphabetically.
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def query_rag(question: str, retriever) -> dict:

    doc_retriver = ConversationalRAG(retriever)
    answer = doc_retriver.invoke(question, chat_history=[])
    context = doc_retriver.get_retrieved_context(question, k=5)  # Retrieve the top 5 relevant documents for the question
    return {"answer": answer, "context": context}



def load_local_dataset(dataset_path: str) -> list[dict]:   # seprates test cases and metadata and returns only the test cases
    """
    Load evaluation questions and expected answers
    from a local JSON dataset.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports the recommended structure:
    #
    # {
    #     "metadata": {...},
    #     "test_cases": [...]
    # }

    test_cases = data.get("test_cases", [])

    if not test_cases:
        raise ValueError(
            "No test cases found in evaluation dataset."
        )

    return test_cases


def main():
    # Load env locally like in the notebook 
    if os.getenv("ENV", "local").lower() != "production":
        load_dotenv()
        log.info("Running in LOCAL mode: .env loaded")

    # 1) Build or load FAISS index from the specified directory
    data_dir = Path(DEEPEVAL_INPUT_DIR)          
    assert data_dir.exists(), f"Input dir not found: {data_dir}"
    paths = list_supported_files(data_dir)  # List all supported files in the input directory
    if not paths:
        log.error("No supported files found in input directory", dir=str(data_dir))
        print("No supported files found in input directory.")
        sys.exit(1)

    # Ingest and index
    chat_ingestor = ChatIngestor(temp_base=UPLOAD_BASE, faiss_base=FAISS_BASE,use_session_dirs = True,session_id= None)
    adapters = [LocalFileAdapter(str(p)) for p in paths]
    retriver =  chat_ingestor.create_retrivel(adapters,chunk_size= 1000,chunk_overlap= 200,k= 5)
    log.info("Ingestion complete", session_id=chat_ingestor.session_id)

    # 2) Pull dataset from Confident AI (cloud service for deepeval)
    
    # dataset = EvaluationDataset()
    # dataset.pull(alias=DATASET_ALIAS)

    # -------------------------------------------------
    # 3) Load local evaluation dataset
    # -------------------------------------------------

    local_dataset = load_local_dataset(
        DEEPEVAL_DATASET_PATH
    )

    log.info(
        "Local evaluation dataset loaded",
        total_test_cases=len(local_dataset)
    )

    deepeval_test_cases = []
    for item in local_dataset:

        question = item.get("input")
        expected_output = item.get("expected_output")

        if not question or not expected_output:

            log.error(
                "Invalid dataset item",
                item=item
            )

            continue

        try:

            result = query_rag(
                question,
                retriver
            )

            # Ensure context is a list
            retrieved_context = result["context"]

            if isinstance(retrieved_context, str):

                retrieved_context = [
                    retrieved_context
                ]

            test_case = LLMTestCase(

                input=question,

                actual_output=result["answer"],

                expected_output=expected_output,

                retrieval_context=retrieved_context,

                context=retrieved_context,
            )

            deepeval_test_cases.append(
                test_case
            )

            log.info(
                "Test case created",
                question=question
            )

        except Exception as e:

            log.error(
                "Failed to build test case",
                error=str(e),
                question=question,
            )



    # -------------------------------------------------
    # 5) Check test cases
    # -------------------------------------------------

    if not deepeval_test_cases:

        raise RuntimeError(
            "No DeepEval test cases were created."
        )
    

    # 4) Evaluate with all metrics
    metrics = [
        AnswerRelevancyMetric(),
        FaithfulnessMetric(),
        ContextualPrecisionMetric(),
        ContextualRecallMetric(),
        ContextualRelevancyMetric(),
        HallucinationMetric(),
    ]

    evaluate(
        test_cases=deepeval_test_cases,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()