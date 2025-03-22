from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import json
import pandas as pd
import os
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('rag_setup')

def setup_rag_system(data_path='cleaned_hotel_bookings.csv', analytics_path='precomputed_analytics.json'):
    """
    Set up the Retrieval-Augmented Generation (RAG) system
    
    Args:
        data_path: Path to the cleaned data CSV
        analytics_path: Path to the precomputed analytics JSON
    """
    logger.info("Starting RAG system setup...")
    
    logger.info(f"Loading analytics from {analytics_path}")
    try:
        with open(analytics_path, 'r') as f:
            analytics_data = json.load(f)
        logger.info("Successfully loaded analytics data")
    except Exception as e:
        logger.error(f"Error loading analytics: {str(e)}")
        analytics_data = {}

    logger.info(f"Loading cleaned data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data with {len(df)} records")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    logger.info("Preparing text representations for embedding")
    df['text'] = df.apply(
        lambda row: (
            f"Hotel: {row['hotel']}, "
            f"Canceled: {row['is_canceled']}, "
            f"Country: {row['country']}, "
            f"ADR: {row['adr']}, "
            f"Lead Time: {row['lead_time']}, "
            f"Reservation Status Date: {row['reservation_status_date']}, "
            f"Adults: {row['adults']}, "
            f"Children: {row.get('children', 0)}, "
            f"Stays Weekend Nights: {row['stays_in_weekend_nights']}, "
            f"Stays Week Nights: {row['stays_in_week_nights']}, "
            f"Meal: {row.get('meal', 'Unknown')}, "
            f"Market Segment: {row.get('market_segment', 'Unknown')}, "
            f"Distribution Channel: {row.get('distribution_channel', 'Unknown')}"
        ), 
        axis=1
    )
    

    logger.info("Initializing sentence transformer model")
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing sentence transformer: {str(e)}")
        raise
    
    # Generate embeddings
    logger.info("Generating embeddings for text data (this may take some time)...")
    try:
        embeddings = embedder.encode(df['text'].tolist(), show_progress_bar=True)
        logger.info(f"Successfully generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise
    
    # Create FAISS index
    logger.info("Creating FAISS index")
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        logger.info(f"FAISS index created with {index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise
    

    os.makedirs('models', exist_ok=True)
    

    logger.info("Saving FAISS index and processed data")
    try:
        faiss.write_index(index, 'hotel_index.faiss')
        df.to_pickle('hotel_data.pkl')
        logger.info("Successfully saved index and data")
    except Exception as e:
        logger.error(f"Error saving index and data: {str(e)}")
        raise

    logger.info("Loading question-answering model")
    try:
        qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
        logger.info("Question-answering model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading QA model: {str(e)}")
        raise
    #sample question
    logger.info("Testing RAG system with example questions")
    test_questions = [
        "Show me total revenue for July 2017",
        "Which locations had the highest booking cancellations?",
        "What is the average price of a hotel booking?",
        "What bookings were canceled on May 6, 2015?",
        "What is the cancellation rate?"
    ]
    
    for question in test_questions:
        try:
            logger.info(f"Testing question: '{question}'")
            answer = test_answer_question(question, df, analytics_data, embedder, index, qa_model)
            logger.info(f"Answer: {answer}")
        except Exception as e:
            logger.error(f"Error processing test question: {str(e)}")
    
    logger.info("RAG system setup complete!")
    return index, embedder, qa_model

def test_answer_question(question, df, analytics_data, embedder, index, qa_model):
    """Test the question answering functionality"""

    question_embedding = embedder.encode([question])
    D, I = index.search(np.array(question_embedding), k=5)
    context = " ".join(df.iloc[I[0]]['text'].tolist())
    analytics_context = json.dumps(analytics_data, indent=2)
    full_context = context + " " + analytics_context

    try:
        result = qa_model(question=question, context=full_context)
        return result['answer']
    except Exception as e:
        return f"Error generating answer: {str(e)}"

if __name__ == "__main__":
    setup_rag_system()