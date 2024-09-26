from abc import ABC, abstractmethod
from pymongo import MongoClient
from datetime import datetime

# Abstract Interface for MongoDB operations
class MongoDBInterface(ABC):

    @abstractmethod
    def insert_query(self, query: str) -> None:
        pass

# Concrete Implementation of MongoDBInterface
class MongoDBManager(MongoDBInterface):
    def __init__(self):
        # Initialize the MongoDB client
        self.client = MongoClient('MongoDBconnection')
        self.db = self.client['NameOfDatabase']
        self.queries_collection = self.db['NameOfCollection']

    # Implementation of insert_query method
    def insert_query(self, query: str) -> None:
        self.queries_collection.insert_one({
            "query": query,
            "timestamp": datetime.now()
        })