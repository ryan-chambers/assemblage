from wine_guide import index_name
from pc_client import create_pinecone_client

pc = create_pinecone_client()

pc.delete_index(name=index_name)
