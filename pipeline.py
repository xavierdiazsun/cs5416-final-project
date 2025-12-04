import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import sqlite3
import pickle
import requests
import werkzeug
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue
import threading

# Read environment variables
TOTAL_NODES = int(os.environ.get('TOTAL_NODES', 1))
NODE_NUMBER = int(os.environ.get('NODE_NUMBER', 0))
NODE_0_IP_PORT = os.environ.get('NODE_0_IP', 'localhost:8000')
NODE_1_IP_PORT = os.environ.get('NODE_1_IP', 'localhost:8000')
NODE_2_IP_PORT = os.environ.get('NODE_2_IP', 'localhost:8000')
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', 'faiss_index.bin')
DOCUMENTS_DIR = os.environ.get('DOCUMENTS_DIR', 'documents/')

NODE_0_IP = NODE_0_IP_PORT.split(':')[0]
NODE_0_PORT = int(NODE_0_IP_PORT.split(':')[1]) if ':' in NODE_0_IP_PORT else 8000
NODE_1_IP = NODE_1_IP_PORT.split(':')[0]
NODE_1_PORT = int(NODE_1_IP_PORT.split(':')[1]) if ':' in NODE_1_IP_PORT else 8000
NODE_2_IP = NODE_2_IP_PORT.split(':')[0]
NODE_2_PORT = int(NODE_2_IP_PORT.split(':')[1]) if ':' in NODE_2_IP_PORT else 8000

# Configuration
CONFIG = {
    'faiss_index_path': FAISS_INDEX_PATH,
    'documents_path': DOCUMENTS_DIR,
    'faiss_dim': 768, #You must use this dimension
    'max_tokens': 128, #You must use this max token limit
    'retrieval_k': 10, #You must retrieve this many documents from the FAISS index
    'truncate_length': 512 # You must use this truncate length
}

# Flask app
app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()

@dataclass
class PipelineRequest:
    request_ids: Optional[List[str]]
    request_id: str
    query: str
    timestamp: float

@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str

class BasePipeline(ABC):
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")
        
        # Model names
        self.embedding_model_name = 'BAAI/bge-base-en-v1.5'
        self.reranker_model_name = 'BAAI/bge-reranker-base'
        self.llm_model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
        self.sentiment_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
        self.safety_model_name = 'unitary/toxic-bert'

        self.session = requests.Session()

    @abstractmethod
    def process_request(self, req: PipelineRequest) -> PipelineResponse:
        pass

    @abstractmethod
    def process_batch(self, reqs: List[PipelineRequest]) -> List[PipelineResponse]:
        """
        Main pipeline execution for a batch of requests.
        """
        pass

    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Step 2: Generate embeddings for a batch of queries"""
        model = SentenceTransformer(self.embedding_model_name).to(self.device)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        del model
        gc.collect()
        return embeddings
    
    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 3: Perform FAISS ANN search for a batch of embeddings"""
        if not os.path.exists(CONFIG['faiss_index_path']):
            raise FileNotFoundError("FAISS index not found. Please create the index before running the pipeline.")
        
        print("Loading FAISS index")
        index = faiss.read_index(CONFIG['faiss_index_path'])
        query_embeddings = query_embeddings.astype('float32')
        _, indices = index.search(query_embeddings, CONFIG['retrieval_k'])
        del index
        gc.collect()
        return [row.tolist() for row in indices]
    
    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        'doc_id': result[0],
                        'title': result[1],
                        'content': result[2],
                        'category': result[3]
                    })
            documents_batch.append(documents)
        conn.close()
        return documents_batch
    
    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Step 5: Rerank retrieved documents for each query in the batch"""
        tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device)
        model.eval()
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc['content']] for doc in documents]
            with torch.no_grad():
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        del model, tokenizer
        gc.collect()
        return reranked_batches
    
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            dtype=torch.float16,
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        del model, tokenizer
        gc.collect()
        return responses
    
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        classifier = hf_pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=self.device
        )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative',
            '2 stars': 'negative',
            '3 stars': 'neutral',
            '4 stars': 'positive',
            '5 stars': 'very positive'
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result['label'], 'neutral'))
        del classifier
        gc.collect()
        return sentiments
    
    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        classifier = hf_pipeline(
            "text-classification",
            model=self.safety_model_name,
            device=self.device
        )
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result['score'] > 0.5)
        del classifier
        gc.collect()
        return toxicity_flags


class Node0Pipeline(BasePipeline):
    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Backwards-compatible single-request entry point that delegates
        to the batch processor with a batch size of 1.
        """
        self.process_batch([request])

    def process_batch(self, reqs: List[PipelineRequest]) -> List[PipelineResponse]:
        if not reqs:
            return []

        batch_size = len(reqs)
        queries = [req.query for req in reqs]

        print("\n" + "="*60)
        print(f"Processing batch of {batch_size} requests")
        print("="*60)
        for req in reqs:
            print(f"- {req.request_id}: {req.query[:50]}...")
        
        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        query_embeddings = self._generate_embeddings_batch(queries)
        data = pickle.dumps({
            "request_ids": [req.request_id for req in reqs],
            "query": {
                "query_embeddings": query_embeddings,
                "queries": queries,
            },
        })
        self.session.post(f"http://{NODE_1_IP_PORT}/query", data=data)

class Node1Pipeline(BasePipeline):
    def process_request(self, req: PipelineRequest) -> PipelineResponse:
        query_ids = req.request_ids
        queries = req.query["queries"]
        query_embeddings = req.query["query_embeddings"]

        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        doc_id_batches = self._faiss_search_batch(query_embeddings)

        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        documents_batch = self._fetch_documents_batch(doc_id_batches)

        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        reranked_docs_batch = self._rerank_documents_batch(
            queries,
            documents_batch
        )

        data = pickle.dumps({
            "request_ids": query_ids,
            "query": {
                "queries": queries,
                "reranked_docs_batch": reranked_docs_batch
            },
        })
        self.session.post(f"http://{NODE_2_IP_PORT}/query", data=data)

    def process_batch(self, reqs):
        pass


class Node2Pipeline(BasePipeline):
    def process_request(self, req: PipelineRequest) -> PipelineResponse:
        query_ids = req.request_ids
        queries = req.query["queries"]
        reranked_docs_batch = req.query["reranked_docs_batch"]

        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        responses_text = self._generate_responses_batch(
            queries,
            reranked_docs_batch
        )

        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        sentiments = self._analyze_sentiment_batch(responses_text)

        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        toxicity_flags = self._filter_response_safety_batch(responses_text)

        data = {
            "request_ids": query_ids,
            "responses_text": responses_text,
            "sentiments": sentiments,
            "toxicity_flags": toxicity_flags,
        }

        self.session.post(f"http://{NODE_0_IP_PORT}/return", json=data)

    def process_batch(self, reqs):
        pass

# Global pipeline instance
pipeline = None

def process_requests_worker():
    """Worker thread that processes requests from the queue"""
    global pipeline
    while True:
        try:
            request_data = request_queue.get()
            if request_data is None:  # Shutdown signal
                break
            
            # Create request object
            req = PipelineRequest(
                request_ids=request_data.get('request_ids'),
                request_id=request_data['request_id'],
                query=request_data['query'],
                timestamp=time.time()
            )
            
            # Process request
            pipeline.process_request(req)
            
            request_queue.task_done()
        except Exception as e:
            print(f"Error processing request: {e}")
            request_queue.task_done()


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle incoming query requests"""
    try:
        try:
            data = request.json
        except werkzeug.exceptions.UnsupportedMediaType:
            data = pickle.loads(request.data)
        request_id = data.get('request_id')
        # Only used for node 0 to node 1 and node 1 to node 2 communication
        request_ids = data.get('request_ids')
        # Use the first request_id if multiple are provided as an identifier for results cache
        if request_ids is not None:
            request_id = request_ids[0]
        query = data.get('query')
        
        if request_id is None or query is None:
            return jsonify({'error': 'Missing request_id or query'}), 400
        
        # Check if result already exists (request already processed)
        with results_lock:
            if request_id in results:
                return jsonify(results[request_id]), 200
        
        print(f"queueing request {request_id}")
        # Add to queue
        request_queue.put({
            'request_ids': request_ids,
            'request_id': request_id,
            'query': query
        })
        
        if NODE_NUMBER != 0:
            return '', 204

        # Wait for processing (with timeout). Very inefficient - would suggest using a more efficient waiting and timeout mechanism.
        timeout = 300  # 5 minutes
        start_wait = time.time()
        while True:
            with results_lock:
                if request_id in results:
                    result = results.pop(request_id)
                    return jsonify(result), 200
            
            if time.time() - start_wait > timeout:
                return jsonify({'error': 'Request timeout'}), 504
            
            time.sleep(0.1)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/return', methods=['POST'])
def handle_return():
    """Only used for node 0 to receive results from node 2"""
    try:
        data = request.json
        request_ids = data['request_ids']
        responses_text = data['responses_text']
        sentiments = data['sentiments']
        toxicity_flags = data['toxicity_flags']

        responses = []
        for idx, request_id in enumerate(request_ids):
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(PipelineResponse(
                request_id=request_id,
                generated_response=responses_text[idx],
                sentiment=sentiments[idx],
                is_toxic=sensitivity_result,
            ))

            with results_lock:
                results[request_id] = {
                    'request_id': request_id,
                    'generated_response': responses_text[idx],
                    'sentiment': sentiments[idx],
                    'is_toxic': sensitivity_result
                }

        return '', 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200


def main():
    """
    Main execution function
    """
    global pipeline
    
    print("="*60)
    print("CUSTOMER SUPPORT PIPELINE")
    print("="*60)
    print(f"\nRunning on Node {NODE_NUMBER} of {TOTAL_NODES} nodes")
    print(f"Node IPs: 0={NODE_0_IP_PORT}, 1={NODE_1_IP_PORT}, 2={NODE_2_IP_PORT}")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    if NODE_NUMBER == 0:
        pipeline = Node0Pipeline()
    elif NODE_NUMBER == 1:
        pipeline = Node1Pipeline()
    else:
        pipeline = Node2Pipeline()
    print("Pipeline initialized!")
    
    # Start worker thread
    worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
    worker_thread.start()
    print("Worker threads started!")
    
    # Start Flask server
    print(f"\nStarting Flask server")
    hostname = "0.0.0.0"
    port = NODE_0_PORT
    if NODE_NUMBER == 1:
        port = NODE_1_PORT
    elif NODE_NUMBER == 2:
        port = NODE_2_PORT
    app.run(host=hostname, port=port, threaded=True)


if __name__ == "__main__":
    main()
