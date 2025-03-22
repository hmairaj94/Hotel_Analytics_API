# Hotel Booking Analytics & QA System

## Overview

This project is an **LLM-Powered Booking Analytics & QA System** developed as per the assignment requirements. It processes hotel booking data, generates analytics, and provides a retrieval-augmented question-answering (RAG) system via a REST API. The system uses FastAPI, FAISS for vector storage, and transformer models to enable natural language queries about hotel bookings. The implementation leverages the Kaggle dataset `hotel_bookings.csv` (cleaned and processed) and includes bonus features like query history tracking and a health check endpoint.

### Assignment Requirements & Implementation Status

#### 1. Data Collection & Preprocessing
- **Requirement**: Use a sample dataset (e.g., Kaggle Dataset), clean it, and store it in a structured format.
- **Implemented**: 
  - Used `hotel_bookings.csv` from Kaggle (assumed based on structure).
  - Preprocessing handled in `data_preprocessing.ipynb`: filled missing values (`children`, `babies`, `country`, etc.), converted dates, and saved as `cleaned_hotel_bookings.csv`.
  - Stored processed data as `hotel_data.pkl` for API usage.

#### 2. Analytics & Reporting
- **Requirement**: Generate revenue trends, cancellation rate, geographical distribution, lead time distribution, and additional analytics using Python tools.
- **Implemented**:
  - **Revenue Trends**: Monthly revenue over time in `analytics_generator.py` and stored in `precomputed_analytics.json`.
  - **Cancellation Rate**: Computed as a percentage of total bookings.
  - **Geographical Distribution**: Cancellations by country (top countries listed).
  - **Lead Time Distribution**: Stats and visualization included.
  - **Additional Analytics**: Average Daily Rate (ADR) by hotel type, stay duration distribution.
  - Tools used: `pandas`, `numpy`, `matplotlib`, `seaborn`.

#### 3. Retrieval-Augmented Question Answering (RAG)
- **Requirement**: Use a vector database (e.g., FAISS) and an open-source LLM for Q&A.
- **Implemented**:
  - Vector database: FAISS (`hotel_index.faiss`) for storing embeddings of booking data.
  - LLM: `distilbert-base-uncased-distilled-squad` (from Hugging Face) for question answering, integrated with RAG in `main.py`.
  - Embeddings generated using `SentenceTransformer('all-MiniLM-L6-v2')` in `rag_setup.py`.
  - Supported example questions:
    - "Show me total revenue for July 2017."
    - "Which locations had the highest booking cancellations?"
    - "What is the average price of a hotel booking?"

#### 4. API Development
- **Requirement**: Build a REST API with `/analytics` and `/ask` endpoints.
- **Implemented**:
  - Framework: FastAPI (`main.py`).
  - **POST /analytics**: Returns precomputed analytics.
  - **POST /ask**: Answers natural language questions about bookings.

#### 5. Performance Evaluation
- **Requirement**: Evaluate Q&A accuracy and API response time.
- **Implemented**:
  - Basic testing in `rag_setup.py` with sample questions (logged results).
  - No formal accuracy metrics or optimization yet (see Limitations).

#### 6. Deployment & Submission
- **Requirement**: Package with README, GitHub repo, sample queries, and a short report.
- **Implemented**:
  - GitHub repo: `https://github.com/hmairaj94/Hotel_Analytics_API.git`.
  - README: This file.
  - Sample queries: Included below.
  - Report: See "Implementation Choices & Challenges" section.

#### Bonus Features
- **Real-time Data Updates**: Not implemented (see Future Improvements).
- **Query History Tracking**: Implemented using SQLite (`query_history.db`) with `/query-history` endpoint.
- **Health Check**: `GET /health` endpoint checks system status (data loading, embeddings, etc.).

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git and Git LFS installed (`git lfs install`)
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Cloning the Repository
1. Clone the repo:
   ```bash
   git clone https://github.com/hmairaj94/Hotel_Analytics_API.git
   cd Hotel_Analytics_API
   ```
2. Pull large files tracked by Git LFS:
   ```bash
   git lfs pull
   ```
   - This downloads `hotel_index.faiss` and `hotel_data.pkl`.

### Running the System
1. **Prepare the Data** (if starting from raw data):
   - Download `hotel_bookings.csv` from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
   - Run `data_preprocessing.ipynb` to generate `cleaned_hotel_bookings.csv`.

2. **Generate Analytics**:
   ```bash
   python analytics_generator.py
   ```
   - Outputs `precomputed_analytics.json` and plots in `visualizations/`.

3. **Set Up RAG** (if regenerating):
   ```bash
   python rag_setup.py
   ```
   - Generates `hotel_data.pkl` and `hotel_index.faiss`.

4. **Start the API**:
   ```bash
   uvicorn main:app --reload
   ```
   - API runs at `http://127.0.0.1:8000`.

---

## Usage

### API Endpoints
- **POST /ask**
  - Request: `{"question": "Your question"}`
  - Example: 
    ```bash
    curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "Show me total revenue for July 2017"}'
    ```
  - Response: `{"question": "...", "answer": "Total revenue for July 2017: $1736787.49"}`

- **POST /analytics**
  - Example: `curl -X POST "http://127.0.0.1:8000/analytics"`
  - Response: JSON with all analytics.

- **GET /health**
  - Example: `curl "http://127.0.0.1:8000/health"`
  - Response: `{"status": "healthy", "components": {...}, "timestamp": "..."}`

- **GET /query-history**
  - Example: `curl "http://127.0.0.1:8000/query-history?limit=5"`
  - Response: `{"history": [...]}`

### Sample Queries & Expected Answers
| Query                                      | Expected Answer                              |
|--------------------------------------------|----------------------------------------------|
| "Show me total revenue for July 2017"      | "Total revenue for July 2017: $1736787.49"  |
| "Which locations had the highest booking cancellations?" | "Countries with highest booking cancellations: PRT: 27519, GBR: 2453, ..." |
| "What is the average price of a hotel booking?" | "The average price of a hotel booking is $101.83" |

---

## Implementation Choices & Challenges

### Choices
- **FastAPI**: Chosen for its async support and automatic Swagger UI.
- **FAISS**: Used for its simplicity and efficiency in vector similarity search.
- **DistilBERT**: Selected as a lightweight, open-source LLM for Q&A.
- **SQLite**: Used for query history due to its simplicity and no external setup.

### Challenges
- **Large File Management**: `hotel_index.faiss` (174 MB) exceeded GitHub's 100 MB limit; resolved with Git LFS.
- **RAG Accuracy**: Limited context size and embedding quality sometimes led to incomplete answers.
- **Performance**: Initial API response times were slow due to model loading; pre-loading models helped.

---

## Limitations
- No real-time data updates (static dataset).
- RAG accuracy varies with question complexity.
- No formal performance evaluation metrics.
