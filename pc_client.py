from pinecone.grpc import PineconeGRPC
from pinecone import Pinecone
from config import AppConfig

def create_pinecone_client(prod: bool = False):
    if prod:
        pc = Pinecone(api_key=AppConfig.pinecone_api_key)
    else:
        pc = PineconeGRPC(
            api_key="pclocal", 
            host="http://localhost:5080" 
        )

    return pc
