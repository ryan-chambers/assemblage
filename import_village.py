import argparse
import uuid
from wine_guide import index_name, embedList, DocumentNotes
from pc_client import create_pinecone_client, create_index_if_not_exists
from pinecone.grpc import GRPCClientConfig

from parse_utils import strip_and_fix_chars

pc = create_pinecone_client()

def parse_document(chunk: str) -> DocumentNotes:
    chunk = strip_and_fix_chars(chunk)
    lines = chunk.splitlines()
    entity_name = lines[0].strip()
    notes = [line[2:].rstrip() for line in lines[1:] if line.startswith('- ')]
    doc = DocumentNotes(name=entity_name, notes=notes, raw=chunk)
    print(f"Parsed document: {doc}")

    return doc

def read_documents(input_file_name: str) -> list[DocumentNotes]:
    """Read a text file and return cleaned chunks of document notes."""
    with open(input_file_name, 'r') as file:
        file_content = file.read()
    all_chunks = file_content.split('\n\n')
    documents = list(map(parse_document, all_chunks))
    return documents

def map_document(document: DocumentNotes) -> str:
    return document.consolidated_note()

def process_document_file(input_file_name: str, data_type: str, batch_size: int = 100):
    """Process a document file and upsert embeddings into the Pinecone index.
    
    Args:
        input_file_name: Path to the input file containing document notes.
        data_type: Type of document (e.g., 'village', 'vineyard').
        batch_size: Number of vectors to upsert in each batch.
    """
    documents = read_documents(input_file_name)
    chunks = [map_document(d) for d in documents]
    embeddings = embedList(chunks)
    index = pc.Index(index_name, grpc_config=GRPCClientConfig(secure=False))
    vectors_batch = []
    for i, embedding in enumerate(embeddings):
        chunk_text = chunks[i]
        deterministic_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{data_type}::{chunk_text}").hex
        vec = {
            'id': deterministic_id,
            'values': embedding,
            'metadata': {'chunk': chunk_text, 'type': data_type, 'name': documents[i].name}
        }
        vectors_batch.append(vec)
        if len(vectors_batch) >= batch_size:
            index.upsert(vectors=vectors_batch)
            vectors_batch = []
    if vectors_batch:
        index.upsert(vectors=vectors_batch)
    print(f"{data_type} file processed and upserted {len(chunks)} results to Pinecone index in batches (batch_size={batch_size}).")


def main():
    parser = argparse.ArgumentParser(description="Process village notes and upsert to Pinecone index.")
    parser.add_argument("input_file", help="Path to the input file containing village notes.")
    args = parser.parse_args()

    print(f"Input file: {args.input_file}")

    create_index_if_not_exists(pc)

    process_document_file(args.input_file, "village")

if __name__ == "__main__":
    main()
