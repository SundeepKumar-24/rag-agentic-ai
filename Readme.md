RAG Chatbot for Agentic AI eBook
1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions based only on the content of the provided Agentic AI eBook.

The system retrieves relevant information from the eBook using vector similarity search and generates grounded answers using an LLM. If the required information is not available in the document, the system refuses to answer.

2. Architecture
Agentic AI eBook (PDF)
        ↓
   PDF Extraction
        ↓
     Chunking
        ↓
OpenAI Embeddings
(text-embedding-3-small)
        ↓
     Pinecone
  Vector Database
        ↓
     User Query
        ↓
OpenAI Embedding
        ↓
Similarity Search
        ↓
Top Relevant Chunks
        ↓
     LangGraph
        ↓
   LLM Generation
    (gpt-4o-mini)
        ↓
Final Answer + Context + Confidence
3. Technologies Used
Python
LangGraph
LangChain
FastAPI
Pinecone
OpenAI API
PyPDF
Pydantic
Requests
Models
Embeddings: text-embedding-3-small
LLM: gpt-4o-mini
4. Project Structure
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── graph.py
│   ├── config.py
│   └── errors.py
│
├── app.py
├── tests_sample_queries.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
5. Prerequisites
Python 3.10 or higher
Git
VS Code
OpenAI API key with available API credits
Pinecone API key
6. Environment Variables

Create a .env file:

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

Do not commit the .env file to GitHub.

7. Installation

Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
8. Document Ingestion

Run:

python -m src.ingestion

The ingestion pipeline:

Loads the Agentic AI eBook PDF.
Extracts text from the PDF.
Splits the text into chunks.
Generates embeddings using text-embedding-3-small.
Stores the embeddings and document chunks in Pinecone.
9. Running the API

Start the FastAPI server:

uvicorn app:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger API documentation:

http://127.0.0.1:8000/docs
10. API Endpoints
Health Check
GET /health

Response:

{
  "status": "ok"
}
Chat
POST /chat

Request:

{
  "query": "What is Agentic AI?"
}

Response contains:

{
  "final_answer": "...",
  "retrieved_context": [],
  "confidence_score": 0.0
}
11. Grounding and Refusal Behavior

The chatbot is designed to answer only using information retrieved from the provided eBook.

If the retrieved information is not sufficiently relevant, the system does not send the query to the LLM and returns:

The information is not available in the provided document.

This prevents the chatbot from generating answers based on information outside the provided document.

12. Confidence Score

The system calculates a confidence/relevance score using the similarity scores of the retrieved chunks.

The score considers:

Best retrieved chunk similarity
Average similarity of retrieved chunks

Refused queries receive a confidence score of 0.0.

13. Testing

Run:

python tests_sample_queries.py

The test script evaluates questions related to the eBook as well as an unrelated question.

Example questions:

What is Agentic AI according to the eBook?
How do AI agents differ from traditional automation systems?
What are the core components of an Agentic Architecture?
What role does memory play in Agentic AI workflows?
Who won the 2022 FIFA World Cup?
What use cases or applications of Agentic AI does the eBook describe?

The results are saved to:

sample_results.json
14. Benchmark Results
Query	Expected	Result
What is Agentic AI according to the eBook?	Answer	TBD
How do AI agents differ from traditional automation systems?	Answer	TBD
What are the core components of an Agentic Architecture?	Answer	TBD
What role does memory play in Agentic AI workflows?	Answer	TBD
Who won the 2022 FIFA World Cup?	Refuse	TBD
What use cases or applications of Agentic AI does the eBook describe?	Answer	TBD

Benchmark values should be updated after successfully running tests_sample_queries.py.

15. Error Handling

The API handles common failures including:

Invalid or empty queries
Missing configuration
Pinecone connection errors
OpenAI API errors
Unexpected application errors

Appropriate HTTP status codes are returned for these cases.

16. Conclusion

This project demonstrates a document-grounded RAG chatbot using LangGraph, Pinecone, OpenAI embeddings, and FastAPI. The system retrieves relevant eBook content before generating an answer and includes refusal behavior to prevent responses that are not supported by the provided document.