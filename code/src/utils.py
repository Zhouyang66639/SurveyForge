import math
from typing import List
import tiktoken
import json
import numpy as np
import pandas as pd
from datetime import timedelta
import faiss

from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore


class tokenCounter():

    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.model_price = {}
        
    def num_tokens_from_string(self, string:str) -> int:
        return len(self.encoding.encode(string))

    def num_tokens_from_list_string(self, list_of_string:List[str]) -> int:
        num = 0
        for s in list_of_string:
            num += len(self.encoding.encode(s))
        return num
    
    def compute_price(self, input_tokens, output_tokens, model):
        return (input_tokens/1000) * self.model_price[model][0] + (output_tokens/1000) * self.model_price[model][1]

    def text_truncation(self,text, max_len = 1000):
        encoded_id = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(encoded_id[:min(max_len,len(encoded_id))])

def autosurvey_db_json2doc_langchain(json_path):
    """
        origion json file: {'cs_paper_info':{'1':{}, '2':{}, ...}}
        step 1: transfer each paper's value to Document class 
                {'cs_paper_info':{'1': Document(), '2': Document(), ...}}
        step 2: store all Document in InMemoryDocstore
    
        Args:
            json_path: path to the json file
        Returns:
            doc_list: list of Document, for Retriever
            doc_store: InMemoryDocstore(dict of Document), for Vectorstore
            index2id: dict of index to id {0: '1', 1: '2', ...}
    """
    doc_list = []
    with open(json_path, 'r') as f:
        doc_db = json.load(f)
    doc_dict_db = {}
    for doc_id, doc_dict in doc_db['cs_paper_info'].items():
        content = doc_dict['abs']
        doc_dict.pop('abs', None)
        doc_dict_db[doc_dict['id']] = Document(
            page_content=content,
            metadata=doc_dict,
        )
        doc_list.append(
            Document(
                page_content=content, 
                metadata=doc_dict
            )
        )
    
    number_of_docs = len(doc_db['cs_paper_info'])
    index2id = {int(index): str(index+1) for index in range(number_of_docs)}
    
    doc_store = InMemoryDocstore(doc_dict_db)
    return doc_list, doc_store, index2id

def postprocess_results_langchain2id(results):
        """
            Args:
                results: list[list[Document]], list of retrieved documents for each query
            Returns:
                references_titles:
                references_abs:
        """
        references_ids = []
        for result in results:
            references_ids.extend([doc.metadata['id'] for doc in result])
        
        return references_ids


def sort_by_citation(documents, top_k=3):
    # sort the documents by citation_count
    documents = sorted(documents, key=lambda x: x.metadata['citation_count'], reverse=True)
    if top_k > len(documents):
        top_k = len(documents)
        print(f"Only {top_k} documents available.")
    # get the top 3 documents
    top_docs = documents[:top_k]
    return top_docs

def get_time_windows(time_oldest, time_newest, period):
    # Convert strings to Timestamps
    time_oldest = pd.to_datetime(time_oldest)
    time_newest = pd.to_datetime(time_newest)
    
    # List to hold the time windows
    time_windows = []
    
    # Generate the time windows
    current_start = time_oldest
    while current_start < time_newest:
        current_end = current_start + pd.DateOffset(years=period) - timedelta(days=1)
        if current_end > time_newest:
            current_end = time_newest
      
        time_windows.append((current_start, current_end))
        
        # Advance start for the next window
        current_start += pd.DateOffset(years=period)
    
    return time_windows

def sort_by_citation_period(documents, top_k=10, period=2):
    time_oldest = '2012-01-01'
    time_newest = '2024-09-26'
    time_windows = get_time_windows(time_oldest, time_newest, period)
    # ratio = top_k/total_doc, for each period, get top_k*period documents
    total_doc = len(documents)
    if total_doc == 0:
        return []
    ratio = top_k / total_doc
    top_docs = []
    for start, end in time_windows:
        docs_in_period = []
        for doc in documents:
            doc_date = pd.to_datetime(doc.metadata['date'])
            if doc_date >= start and doc_date <= end:
                docs_in_period.append(doc)

        if len(docs_in_period) == 0:
            continue
        top_k_period = math.ceil(ratio*len(docs_in_period))

        selected_docs = sort_by_citation(docs_in_period, top_k_period)
        top_docs.extend(selected_docs)
        
    return top_docs
    
def get_index_filter(arxivid_to_index, results_arxivid):
    # transfer arxivid to index in outline_rag_results
    results_index = [arxivid_to_index[arxivid] for arxivid in results_arxivid]
    results_index_np = np.asarray(results_index, dtype="int64")

    # FAISS python bindings differ by version:
    # - old versions may accept IDSelectorArray(list)
    # - faiss==1.7.x often requires (n, pointer) arguments
    # Prefer IDSelectorBatch with pointer, then fall back.
    try:
        id_selector = faiss.IDSelectorBatch(
            results_index_np.size,
            faiss.swig_ptr(results_index_np),
        )
    except Exception:
        try:
            id_selector = faiss.IDSelectorArray(
                results_index_np.size,
                faiss.swig_ptr(results_index_np),
            )
        except Exception:
            id_selector = faiss.IDSelectorArray(results_index)

    index_filter = {
        'id_selector': id_selector,
    }
    return index_filter
