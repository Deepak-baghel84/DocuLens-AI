import os
import sys
from pathlib import Path
from typing import List
import json
from dotenv import load_dotenv
from groq_eval_model import GroqEvalModel
import time

#from deepeval.dataset import EvaluationDataset
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
from collections import defaultdict
from groq_eval_model import GroqEvalModel


eval_model = GroqEvalModel(
    model_name="openai/gpt-oss-120b"
)


        # Directory containing documents to ingest and index(raw data for evaluation)
DEEPEVAL_INPUT_DIR = os.getenv("DEEPEVAL_INPUT_DIR","data_deep_eval")

             # Document versioning path
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "Data/multidoc_archive")    

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")   # faiss index path
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

       # Local JSON evaluation dataset
DEEPEVAL_DATASET_PATH = os.getenv("DEEPEVAL_DATASET_PATH", "eval_ques/omnibench_rag_evaluation_dataset_formatted.json")  # dataset path, questions and expected answers for evaluation













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



def _log_docs(docs):

    print("\n" + "=" * 80)
    print(f"RETRIEVED {len(docs)} DOCUMENTS")
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):

        print(f"\n--- DOCUMENT {i} ---")

        print("METADATA:")
        print(doc.metadata)

        print("\nCONTENT:")
        print(doc.page_content[:1000])

        print("\n" + "-" * 80)

    return 




def query_rag(question: str,rag: ConversationalRAG) -> dict:

    result = rag.invoke_with_context(
        question=question,
        chat_history=[]
    )

    documents = result["documents"]

    context = [doc.page_content for doc in documents]

    print(f"RETRIEVED {len(documents)} DOCUMENTS")

   # _log_docs(documents)  # Log the retrieved documents

    return {
        "answer": result["answer"],
        "context": context
    }



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
    paths = list_supported_files(data_dir)  # List all supported files in the input directory(raw data)
    if not paths:
        log.error("No supported files found in input directory", dir=str(data_dir))
        print("No supported files found in input directory.")
        sys.exit(1)

    # Ingest and index
    chat_ingestor = ChatIngestor(temp_base=UPLOAD_BASE, faiss_base=FAISS_BASE,use_session_dirs = True,session_id= None) #raw dataset
    adapters = [LocalFileAdapter(str(p)) for p in paths]
    retriever =  chat_ingestor.create_retrivel(adapters,chunk_size= 1000,chunk_overlap= 200,k= 5)
    log.info("Ingestion complete", session_id=chat_ingestor.session_id)

    # print("\n" + "=" * 70)
    # print("RAW INPUT FILES")
    # print("=" * 70)

    # print(f"Input directory: {data_dir}")
    # print(f"Total supported files found: {len(paths)}")

    # for i, path in enumerate(paths, start=1):
    #     print(f"\n{i}. File name : {path.name}")
    #     print(f"   Full path : {path}")
    #     print(f"   Extension : {path.suffix}")
    #     print(f"   Exists    : {path.exists()}")
    #     print(f"   Size      : {path.stat().st_size / 1024:.2f} KB")

    # print("=" * 70)

   

    # print("\n========== RAW RETRIEVED DOCUMENTS ==========")

    # for i, doc in enumerate(documents, start=1):
    #     print(f"\n--- DOCUMENT {i} ---")
    #     print("PAGE CONTENT:")
    #     print(repr(doc.page_content[:1000]))
    #     print("\nMETADATA:")
    #     print(doc.metadata)

    rag = ConversationalRAG(
    session_id=chat_ingestor.session_id
)

 #create retrievel only once and use it for all queries

    index_dir = os.path.join(FAISS_BASE, chat_ingestor.session_id)
    rag.load_retriever_from_faiss(index_path=index_dir,k=5,index_name=FAISS_INDEX_NAME) 

      

    # 2) Pull dataset from Confident AI (cloud service for deepeval)
    
    # dataset = EvaluationDataset()
    # dataset.pull(alias=DATASET_ALIAS)

    # -------------------------------------------------
    # 3) Load local evaluation dataset(goldens)
    # -------------------------------------------------

    local_dataset = load_local_dataset(
        DEEPEVAL_DATASET_PATH
    )

    log.info(
        "Local evaluation dataset loaded",
        total_test_cases=len(local_dataset)
    )

    deepeval_test_cases = []

    stop = 0
    for item in local_dataset:

        if stop > 1:
            break
        stop += 1

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
                rag=rag
            )
            log.info("Query executed successfully",
                question=question,
                answer=result["answer"],
                context=result["context"]
            )
            # Ensure context is a list
            retrieved_context = result["context"]

            if isinstance(retrieved_context, str):

                retrieved_context = [
                    retrieved_context
                ]

            # log.info("Retrieved context and test case creation started")

            test_case = LLMTestCase(
                input=question,

                actual_output=result["answer"],

                expected_output=expected_output,

                retrieval_context=retrieved_context,

                context=retrieved_context,             # we considering both retrieval context and context as the same for evaluation purpose
            )

            deepeval_test_cases.append(
                test_case
            )

            log.info(
                "Test case created successfully",
                question=question
            )

        except Exception as e:

            log.error(
                "Failed to build test case",
                error=str(e),
                question=question,
            )


    time.sleep(70)  # wait for a minute to avoid rate limiting issues with the evaluation model

    # -------------------------------------------------
    # 5) Check test cases
    # -------------------------------------------------

    if not deepeval_test_cases:

        raise RuntimeError(
            "No DeepEval test cases were created."
        )
    

    # 4) Evaluate with all metrics
    metrics = [
        AnswerRelevancyMetric(model=eval_model),
        FaithfulnessMetric(model=eval_model),
        ContextualPrecisionMetric(model=eval_model),
        ContextualRecallMetric(model=eval_model),
        ContextualRelevancyMetric(model=eval_model),
        HallucinationMetric(model=eval_model),
    ]

    evaluation_results =evaluate(
        test_cases=deepeval_test_cases,
        metrics=metrics,
    )
    log.info("Evaluation completed", total_test_cases=len(evaluation_results.test_results))


    
    for result in evaluation_results.test_results:


        print("\n" + "=" * 80)

        print("QUESTION:")
        print(result.input)

        print("\nRAG ANSWER:")
        print(result.actual_output)

        print("\nEXPECTED ANSWER:")
        print(result.expected_output)

        print("\nMETRIC RESULTS:")
        print("\nMETRIC RESULTS:")

        for metric_data in result.metrics_data:
            print(f"{metric_data.name}: "f"{metric_data.score}")

        print("=" * 80)


    metric_scores = defaultdict(list)

    for test_result in evaluation_results.test_results:
        for metric in test_result.metrics_data:
            metric_scores[metric.name].append(metric.score)


    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)

    for metric_name, scores in metric_scores.items():
        average_score = sum(scores) / len(scores)

        print(f"{metric_name}: {average_score:.4f}")



if __name__ == "__main__":
    main()



