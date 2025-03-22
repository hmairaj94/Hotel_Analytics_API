from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
from datetime import datetime
import sqlite3
from typing import List, Optional

app = FastAPI(title="Hotel Booking Analytics API", 
              description="A system that processes hotel booking data, extracts insights, and answers queries")

class QuestionRequest(BaseModel):
    question: str

class HealthResponse(BaseModel):
    status: str
    components: dict
    timestamp: str

# Database setup for query history (Bonus feature)
def get_db():
    conn = sqlite3.connect('query_history.db')
    try:
       
        conn.execute('''
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        yield conn
    finally:
        conn.close()


try:
    # Load data
    with open('precomputed_analytics.json', 'r') as f:
        analytics_data = json.load(f)
    df = pd.read_pickle('hotel_data.pkl')
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    index = faiss.read_index('hotel_index.faiss')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    
    components_status = {
        "data_loading": "healthy",
        "embeddings": "healthy",
        "faiss_index": "healthy",
        "qa_model": "healthy"
    }
except Exception as e:
    components_status = {
        "error": str(e)
    }

@app.post("/ask")
def ask_question(request: QuestionRequest, conn: sqlite3.Connection = Depends(get_db)):
    try:
        answer = answer_question(request.question)
        conn.execute(
            "INSERT INTO query_history (question, answer) VALUES (?, ?)",
            (request.question, str(answer))
        )
        conn.commit()
        
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

def answer_question(question):
    print(f"DEBUG: Received question: {question}")
    question_lower = question.lower()
    
    # Override flag to track if we've found an answer
    answer_found = False
    answer_result = None

    # Check for revenue-related questions
    revenue_pattern = r"(?:show me\s+)?total revenue for (?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    revenue_match = re.search(revenue_pattern, question_lower, re.IGNORECASE)
    if "revenue" in question_lower and revenue_match:
        month_year_str = revenue_match.group(0).replace("total revenue for ", "").replace("show me ", "")
        month_map = {
            "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
        }
        month_name, year = month_year_str.split()
        month_key = month_map[month_name.lower()]
        target_key = f"{year}-{month_key}"
        
        print(f"DEBUG: Question matched revenue pattern. Target key: {target_key}")
        
        
        try:
            
            print(f"DEBUG: Looking for revenue data with key: {target_key}")
            print(f"DEBUG: Available keys in revenue_trends: {list(analytics_data['revenue_trends'].keys())}")
            
            if target_key in analytics_data['revenue_trends']:
                revenue = analytics_data['revenue_trends'][target_key]
                answer_result = f"Total revenue for {month_year_str}: ${float(revenue):.2f}"
                answer_found = True
                print(f"DEBUG: Found revenue. Result: {answer_result}")
                
           
            if not answer_found:
                for key in analytics_data['revenue_trends'].keys():
                    if key.startswith(target_key):
                        revenue = analytics_data['revenue_trends'][key]
                        answer_result = f"Total revenue for {month_year_str}: ${float(revenue):.2f}"
                        answer_found = True
                        print(f"DEBUG: Found revenue with partial key. Result: {answer_result}")
                        break
            
         
            if not answer_found:
                answer_result = f"No revenue data found for {month_year_str}. Available months: {', '.join(list(analytics_data['revenue_trends'].keys())[:5])}..."
                answer_found = True
                print(f"DEBUG: No revenue found. Result: {answer_result}")
            
        except (KeyError, TypeError, ValueError) as e:
            
            print(f"DEBUG: Error accessing revenue data: {str(e)}")
            try:
                month_start = f"{year}-{month_key}-01"
                month_end = f"{year}-{month_key}-31"
                
                
                monthly_revenue = df[
                    (df['reservation_status_date'] >= month_start) & 
                    (df['reservation_status_date'] <= month_end) &
                    (df['is_canceled'] == 0)
                ]['adr'].sum()
                
                answer_result = f"Total revenue for {month_year_str} (calculated from raw data): ${monthly_revenue:.2f}"
                answer_found = True
                print(f"DEBUG: Calculated revenue from raw data. Result: {answer_result}")
            except Exception as calc_error:
                print(f"DEBUG: Error calculating from raw data: {str(calc_error)}")
                answer_result = f"Unable to calculate revenue for {month_year_str}: {str(calc_error)}"
                answer_found = True

    if not answer_found:
        date_pattern = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"
        date_match = re.search(date_pattern, question)
        if "canceled" in question_lower and date_match:
            print(f"DEBUG: Question matched cancellation date pattern")
            try:
                target_date = pd.to_datetime(date_match.group(0), errors='coerce')
                if target_date is not None:
                    canceled_on_date = df[
                        (df['is_canceled'] == 1) & 
                        (df['reservation_status_date'] == target_date)
                    ]
                    if not canceled_on_date.empty:
                        bookings = canceled_on_date[['hotel', 'country', 'adr', 'lead_time']].to_dict(orient='records')
                        answer_result = {"date": target_date.strftime('%Y-%m-%d'), "canceled_bookings": bookings}
                    else:
                        answer_result = {"date": target_date.strftime('%Y-%m-%d'), "canceled_bookings": []}
                    answer_found = True
                    print(f"DEBUG: Found cancellation data. Result: {answer_result}")
            except Exception as e:
                print(f"DEBUG: Error processing cancellation query: {str(e)}")

    if not answer_found:
        if "highest booking cancellations" in question_lower or "most cancellations" in question_lower:
            print(f"DEBUG: Question matched highest cancellations pattern")
            if 'cancellation_by_country' in analytics_data:
                countries = sorted(analytics_data['cancellation_by_country'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
                result = "Countries with highest booking cancellations:\n"
                for country, count in countries:
                    result += f"{country}: {count} cancellations\n"
                answer_result = result
                answer_found = True
                print(f"DEBUG: Found cancellation by country data. Result: {answer_result}")
            else:
                answer_result = "Cancellation data by country not available"
                answer_found = True

    if not answer_found:
        if "average price" in question_lower or "average cost" in question_lower:
            print(f"DEBUG: Question matched average price pattern")
            if 'average_adr' in analytics_data:
                answer_result = f"The average price of a hotel booking is ${analytics_data['average_adr']:.2f}"
                answer_found = True
                print(f"DEBUG: Found average price. Result: {answer_result}")
            else:
                avg_price = df['adr'].mean()
                answer_result = f"The average price of a hotel booking is ${avg_price:.2f}"
                answer_found = True
                print(f"DEBUG: Calculated average price. Result: {answer_result}")

    if answer_found:
        print(f"DEBUG: Returning answer: {answer_result}")
        return answer_result

    print("DEBUG: No pattern matched, falling back to RAG")
    
    try:
        question_embedding = embedder.encode([question])
        D, I = index.search(np.array(question_embedding), k=5)
        context = " ".join(df.iloc[I[0]]['text'].tolist())
        
        analytics_text = "Revenue information:\n"
        if 'revenue_trends' in analytics_data:
            for month, amount in list(analytics_data['revenue_trends'].items())[:5]:
                analytics_text += f"- {month}: ${float(amount):.2f}\n"
        analytics_text += f"Average price: ${analytics_data.get('average_adr', 0):.2f}\n"
        analytics_text += f"Cancellation rate: {analytics_data.get('cancellation_rate', 0):.2f}%\n"
        
        full_context = context + " " + analytics_text
        print(f"DEBUG: Using RAG with context length: {len(full_context)}")
        
        result = qa_model(question=question, context=full_context)
        print(f"DEBUG: RAG returned answer: {result['answer']}")
        return result['answer']
    except Exception as e:
        print(f"DEBUG: Error in RAG fallback: {str(e)}")
        return f"Error processing question: {str(e)}"

@app.post("/analytics")
def get_analytics():
    return analytics_data

@app.get("/health")
def check_health():
    """Health check endpoint to verify system status"""
    all_healthy = all(status == "healthy" for status in components_status.values())
    status = "healthy" if all_healthy else "unhealthy"
    
    return HealthResponse(
        status=status,
        components=components_status,
        timestamp=datetime.now().isoformat()
    )

@app.get("/query-history")
def get_query_history(limit: Optional[int] = 10, conn: sqlite3.Connection = Depends(get_db)):
    """Get history of previous queries (Bonus feature)"""
    cursor = conn.execute("SELECT question, answer, timestamp FROM query_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    history = [{"question": row[0], "answer": row[1], "timestamp": row[2]} for row in cursor.fetchall()]
    return {"history": history}

# Run with: uvicorn main:app --reload