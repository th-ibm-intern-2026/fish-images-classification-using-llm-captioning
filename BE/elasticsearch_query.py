import pandas as pd
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from elasticsearch.helpers import bulk
import os

class ElasticsearchQuery:
    def __init__(self, es_endpoint, es_username, es_password):
        self.es = Elasticsearch(
        [es_endpoint],
        http_auth=(es_username, es_password),
        verify_certs=False
    )
    def list_all_index(self, creator="user"):
        """
        Args:
            creator (str):
                - "user": indices not starting with '.'
                - "system": indices starting with '.'
                - "all": all indices categorized
        """
        try:
            user_index = []
            system_index = []
            indices = self.es.indices.get_alias()
            for index_name in indices.keys():
                count = self.get_document_count(index_name, silent=True)
                if not(index_name.startswith('.')):
                    user_index.append(index_name)
                else:
                    system_index.append(index_name)
            print(f"User Indices: {user_index}")
            if creator == "user":
                return user_index
            elif creator == "system":
                return system_index
            elif creator == "all":
                return indices.keys()
        except Exception as e:
            print(f"✗ Error listing indices: {e}")

    def search_text(self, index_name, field, text, size=10):
        """Search text in specific field"""
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {"match": {field: text}},
                    "size": size
                }
            )
            docs = [hit['_source'] for hit in response['hits']['hits']]
            print(f"📄 Found {len(docs)} matches for '{text}' in {field}")
            return docs
        except Exception as e:
            print(f"✗ Search error: {e}")
    
    def search_exact(self, index_name, field, value, size=10):
        """Search exact match"""
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {"term": {field: value}},
                    "size": size
                }
            )
            docs = [hit['_source'] for hit in response['hits']['hits']]
            print(f"📄 Found {len(docs)} exact matches")
            return docs
        except Exception as e:
            print(f"✗ Search error: {e}")
    
    def search_embedding(self, index_name, embedding_field, query_vector, size=10):
        """Search similar vectors using kNN"""
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "knn": {
                        "field": embedding_field,
                        "query_vector": query_vector,
                        "k": size,
                        "num_candidates": size * 2
                    },
                    "_source": True
                }
            )
            return response
        except Exception as e:
            print(f"✗ Embedding search error: {e}")
    
    def count_docs(self, index_name, query=None):
        """Count documents matching query"""
        try:
            body = {"query": query} if query else {"query": {"match_all": {}}}
            response = self.es.count(index=index_name, body=body)
            count = response['count']
            print(f"📊 Count: {count}")
            return count
        except Exception as e:
            print(f"✗ Count error: {e}")

