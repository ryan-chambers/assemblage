from pinecone.grpc import PineconeGRPC
from pinecone import Pinecone, ServerlessSpec
from config import AppConfig
from wine_guide import index_name

def create_pinecone_client(prod: bool = False):
    if prod:
        pc = Pinecone(api_key=AppConfig.pinecone_api_key)
    else:
        pc = PineconeGRPC(
            api_key="pclocal", 
            host="http://localhost:5080" 
        )

    return pc

def create_index_if_not_exists(pc: Pinecone | PineconeGRPC):
    if not pc.has_index(name=index_name):
        print('Creating index...')
        pc.create_index(
            name=index_name,
            dimension=3072, # dimensionality of text-embedding-large-003
            metric='dotproduct',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        print('Index already exists.')
