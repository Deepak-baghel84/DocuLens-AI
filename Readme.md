# 📄 Document Portal - AI-Powered Document Intelligence System
Overview

Document Portal is a Retrieval-Augmented Generation (RAG) application that enables users to upload documents and interact with them through natural language conversations. Instead of manually searching through lengthy documents, users can ask questions and receive context-aware answers generated using Large Language Models (LLMs).

The system combines semantic search, vector embeddings, and LLM reasoning to provide accurate responses based on document content while minimizing hallucinations


## Features
- Upload and process PDF documents
- Intelligent document question-answering
- Context-aware conversational chat
- Retrieval-Augmented Generation (RAG) pipeline
- Semantic search using vector embeddings
- Multi-turn conversation support
- Chat history-aware query reformulation
- Source-grounded responses
- Fast inference using Groq-hosted LLMs
- Responsive web interface
- Cloud deployment on AWS ECS Fargate


## Architecture
```text
User Query
     │
     ▼
Chat History
     │
     ▼
Question Contextualization
     │
     ▼
Embedding Generation
     │
     ▼
FAISS Vector Store Retrieval
     │
     ▼
Relevant Context Extraction
     │
     ▼
LLM (Groq / Llama)
     │
     ▼
Generated Response
```

## Tech Stack
- Frontend
- Streamlit
- Backend
- Python
- LangChain
- LLM
- Groq API
- Llama Models
- Vector Database
- FAISS
- Document Processing
- PyPDF
- Recursive Character Text Splitter
- Deployment
- Docker
- AWS ECS
- AWS Fargate
- Amazon ECR


## How It Works
1. **Document Ingestion**

The uploaded document is parsed and converted into raw text.

2. **Text Chunking**

The document is divided into manageable chunks while preserving context through chunk overlap.

3. **Embedding Generation**

Each chunk is transformed into vector embeddings.

4. **Vector Storage**

Embeddings are stored in FAISS for efficient similarity search.

5. **Context Retrieval**

When a user asks a question, the system retrieves the most relevant document chunks.

6. **Response Generation**

The retrieved context is supplied to the LLM, which generates an answer grounded in the document content.


## Project Structure
```text
Document-Portal/
│
├── main.py
├── .github/
│     ├── aws.yaml
│     ├── ci.yaml
│     └──task_definition.json
├── Dockerfile
├── src/
│   ├── document_analyzer
│   ├── document_chat
│   ├── document_compare
│   └── document_ingestion
│
├── utils
│     ├── model_utils.py
│     ├── config_util.py
│     ├── file_io.py
│     └── document_ops.py
│    
│── requirements.txt
├── Tests/
├── data/
└── README.md
```

## Installation
**Clone Repository**
```bash
git clone <repository-url>
cd document-portal
Create Virtual Environment
python -m venv venv
Activate Environment
source venv/bin/activate
```
 ## Windows:
```
venv\Scripts\activate
```
**Install Dependencies**
```
pip install -r requirements.txt
```
Configure Environment Variables

**Create a .env file**
```
GROQ_API_KEY=your_api_key
```
**Run Application**
```
uvicorn api.main:app --reload
```

## AWS Deployment

The application is containerized using Docker and deployed on AWS using:

Amazon Elastic Container Registry (ECR)
Amazon ECS
AWS Fargate
Application Load Balancer (ALB)

### Deployment benefits:

Scalable infrastructure
Serverless container management
High availability
Simplified deployment workflow


### Key Learning Outcomes
Retrieval-Augmented Generation (RAG)
Vector Databases and Semantic Search
LangChain Framework
Prompt Engineering
LLM Application Development
Cloud Deployment with AWS
Docker Containerization
Conversational AI Systems


### Future Improvements
- Document source citations
- Hybrid Search (Semantic + Keyword)
- Re-ranking for retrieval quality
- Multi-document chat
- User authentication
- Chat session persistence
- RAG evaluation using Ragas and LangSmith
- Support for Word, PPT, and Excel files


### Results
Enables natural language interaction with documents.
Reduces manual document search effort.
Generates context-aware responses grounded in source documents.
Supports efficient knowledge extraction from large PDFs.

---
### Author

**Deepak Baghel**

B.Tech CSE (AI & ML)

Passionate about Machine Learning, Generative AI, Retrieval-Augmented Generation, and Computer Vision.

LinkedIn: https://www.linkedin.com/in/deepak-baghel-3282a4300/

GitHub: https://github.com/Deepak-baghel84
