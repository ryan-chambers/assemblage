from dotenv import load_dotenv
import os

class Config:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
    
AppConfig = Config()
