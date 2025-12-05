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
from queue import Queue, Empty
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
    'faiss_dim': 768, 
    'max_tokens': 128, 
    'retrieval_k': 10, 
    'truncate_length': 512,
    'batch_size': 8
}

# Flask app
app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()
pending_events = {} 

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # Uses self.embed_model loaded in Node0Pipeline
        embeddings = self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        # Uses self.index loaded in Node1Pipeline
        if self.index is None:
             return [[] for _ in query_embeddings]
             
        query_embeddings = query_embeddings.astype('float32')
        _, indices = self.index.search(query_embeddings, CONFIG['retrieval_k'])
        return [row.tolist() for row in indices]
    
    def _fetch_documents_batch(self, doc_id_batches: List[List[int]]) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        if not os.path.exists(db_path): return [[] for _ in doc_id_batches]
        
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
                inputs = self.rerank_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=CONFIG['truncate_length']
                ).to(self.device)
                scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores][:3])
        return reranked_batches
    
    def _generate_responses_batch(self, queries: List[str], documents_batch: List[List[Dict]]) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join([f"- {doc['title']}: {doc['content'][:200]}" for doc in documents])
            messages = [
                {"role": "system",
                 "content": "When given Context and Question, reply as 'Answer: <final answer>' only."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ]
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
            
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=CONFIG['max_tokens'],
                temperature=0.01,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        return responses
    
    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        # Uses self.sentiment_pipeline loaded in Node2Pipeline
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.sentiment_pipeline(truncated_texts)
        sentiment_map = {
            '1 star': 'very negative', '2 stars': 'negative', '3 stars': 'neutral',
            '4 stars': 'positive', '5 stars': 'very positive'
        }
        sentiments = [sentiment_map.get(r['label'], 'neutral') for r in raw_results]
        return sentiments
    
    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        truncated_texts = [text[:CONFIG['truncate_length']] for text in texts]
        raw_results = self.safety_pipeline(truncated_texts)
        return [r['score'] > 0.5 for r in raw_results]


class Node0Pipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        print("Loading Embedding Model...")
        self.embed_model = SentenceTransformer(self.embedding_model_name).to(self.device)
        self.embed_model.eval()

    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        self.process_batch([request])

    def process_batch(self, reqs: List[PipelineRequest]) -> List[PipelineResponse]:
        if not reqs: return []
        batch_size = len(reqs)
        queries = [req.query for req in reqs]
        print(f"\n[Node 0] Processing batch of {batch_size} requests")
        
        # Step 1: Generate embeddings
        query_embeddings = self._generate_embeddings_batch(queries)
        
        data = pickle.dumps({
            "request_ids": [req.request_id for req in reqs],
            "query": {
                "query_embeddings": query_embeddings,
                "queries": queries,
            },
        })
        try:
            self.session.post(f"http://{NODE_1_IP_PORT}/query", data=data, timeout=1.0)
        except: pass # Async fire and forget

class Node1Pipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        # Load Index ONCE at startup
        print("Loading FAISS Index...")
        if os.path.exists(CONFIG['faiss_index_path']):
            self.index = faiss.read_index(CONFIG['faiss_index_path'])
        else:
            print("WARNING: FAISS Index not found")
            self.index = None
            
        # Load Reranker ONCE at startup
        print("Loading Reranker...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name).to(self.device)
        self.rerank_model.eval()

    def process_request(self, req: PipelineRequest) -> PipelineResponse:
        # Extract batch data from packet
        query_ids = req.request_ids
        queries = req.query["queries"]
        query_embeddings = req.query["query_embeddings"]
        print(f"\n[Node 1] Processing batch of {len(query_ids)}")

        # Step 2: FAISS
        doc_id_batches = self._faiss_search_batch(query_embeddings)
        # Step 3: Fetch
        documents_batch = self._fetch_documents_batch(doc_id_batches)
        # Step 4: Rerank
        reranked_docs_batch = self._rerank_documents_batch(queries, documents_batch)

        data = pickle.dumps({
            "request_ids": query_ids,
            "query": {
                "queries": queries,
                "reranked_docs_batch": reranked_docs_batch
            },
        })
        try:
            self.session.post(f"http://{NODE_2_IP_PORT}/query", data=data, timeout=1.0)
        except: pass

    def process_batch(self, reqs): pass


class Node2Pipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        print("Loading LLM...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)
        
        print("Loading classifiers...")
        self.sentiment_pipeline = hf_pipeline("sentiment-analysis", model=self.sentiment_model_name, device=0 if torch.cuda.is_available() else -1)
        self.safety_pipeline = hf_pipeline("text-classification", model=self.safety_model_name, device=0 if torch.cuda.is_available() else -1)

    def process_request(self, req: PipelineRequest) -> PipelineResponse:
        query_ids = req.request_ids
        queries = req.query["queries"]
        reranked_docs_batch = req.query["reranked_docs_batch"]
        print(f"\n[Node 2] Processing batch of {len(query_ids)}")

        # Step 5: LLM
        responses_text = self._generate_responses_batch(queries, reranked_docs_batch)
        # Step 6: Sentiment
        sentiments = self._analyze_sentiment_batch(responses_text)
        # Step 7: Safety
        toxicity_flags = self._filter_response_safety_batch(responses_text)

        data = {
            "request_ids": query_ids,
            "responses_text": responses_text,
            "sentiments": sentiments,
            "toxicity_flags": toxicity_flags,
        }
        try:
            self.session.post(f"http://{NODE_0_IP_PORT}/return", json=data, timeout=5.0)
        except: pass

    def process_batch(self, reqs): pass

# Global pipeline instance
pipeline = None

def process_requests_worker():
    global pipeline
    while True:
        try:
            # 1. Block for first item
            first_item = request_queue.get(timeout=1.0) 
            batch = [first_item]
            
            # 2. Opportunistic wait for more
            start_wait = time.time()
            while len(batch) < CONFIG['batch_size']:
                if time.time() - start_wait > 0.1: break # Wait max 100ms
                try:
                    item = request_queue.get_nowait()
                    batch.append(item)
                except Empty:
                    time.sleep(0.01)

            # 3. Process the batch
            if NODE_NUMBER == 0:
                # Node 0: Client requests
                req_objs = []
                for item in batch:
                    if item.get('request_ids'): # Handling weird case
                        pass 
                    else:
                        req_objs.append(PipelineRequest(
                            request_ids=[item['request_id']],
                            request_id=item['request_id'],
                            query=item['query'],
                            timestamp=time.time()
                        ))
                if req_objs: pipeline.process_batch(req_objs)
            else:
                # Nodes 1 & 2: Process packets individually (they contain batches)
                for item in batch:
                    req_obj = PipelineRequest(
                        request_ids=item.get('request_ids'),
                        request_id=None,
                        query=item.get('query'),
                        timestamp=time.time()
                    )
                    pipeline.process_request(req_obj)
            
        except Empty:
            continue
        except Exception as e:
            print(f"Worker Error: {e}")

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        try:
            data = request.json
            is_client = True
        except:
            data = pickle.loads(request.data)
            is_client = False
            
        if is_client:
            req_id = data.get('request_id')
            query = data.get('query')
            
            completion_event = threading.Event()
            with results_lock:
                pending_events[req_id] = completion_event
            
            request_queue.put({'request_id': req_id, 'query': query})
            
            flag = completion_event.wait(timeout=300)
            
            if flag:
                with results_lock:
                    result = results.pop(req_id, None)
                    if req_id in pending_events: del pending_events[req_id]
                return jsonify(result), 200
            else:
                with results_lock:
                    if req_id in pending_events: del pending_events[req_id]
                return jsonify({'error': 'Timeout'}), 504
        else:
            request_queue.put(data)
            return '', 204
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/return', methods=['POST'])
def handle_return():
    try:
        data = request.json
        request_ids = data['request_ids']
        responses_text = data['responses_text']
        sentiments = data['sentiments']
        toxicity_flags = data['toxicity_flags']

        with results_lock:
            for idx, request_id in enumerate(request_ids):
                sensitivity_result = "true" if toxicity_flags[idx] else "false"
                results[request_id] = {
                    'request_id': request_id,
                    'generated_response': responses_text[idx],
                    'sentiment': sentiments[idx],
                    'is_toxic': sensitivity_result
                }
                if request_id in pending_events:
                    pending_events[request_id].set()

        return '', 204
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'node': NODE_NUMBER,
        'total_nodes': TOTAL_NODES
    }), 200

def main():
    global pipeline
    print("="*60)
    print(f"STARTING NODE {NODE_NUMBER}")
    print("="*60)
    
    if NODE_NUMBER == 0:
        pipeline = Node0Pipeline()
    elif NODE_NUMBER == 1:
        pipeline = Node1Pipeline()
    else:
        pipeline = Node2Pipeline()
        
    worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
    worker_thread.start()
    
    port = [NODE_0_PORT, NODE_1_PORT, NODE_2_PORT][NODE_NUMBER]
    app.run(host="0.0.0.0", port=port, threaded=True)

if __name__ == "__main__":
    main()