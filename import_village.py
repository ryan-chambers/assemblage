import argparse
import uuid
from wine_guide import index_name, embedList, VillageNotes
from pc_client import create_pinecone_client, create_index_if_not_exists
from pinecone.grpc import GRPCClientConfig

from parse_utils import strip_and_fix_chars

pc = create_pinecone_client()

def parse_village(chunk: str) -> VillageNotes:
    chunk = strip_and_fix_chars(chunk)
    lines = chunk.splitlines()
    village_name = lines[0].strip()
    notes = [line[2:].rstrip() for line in lines[1:] if line.startswith('- ')]
    vn = VillageNotes(village=village_name, notes=notes, raw=chunk)
    print(f"Parsed village: {vn}")

    return vn

def read_villages(input_file_name: str) -> list[VillageNotes]:
    """Read a text file and return cleaned chunks of village notes."""
    with open(input_file_name, 'r') as file:
        file_content = file.read()
    all_chunks = file_content.split('\n\n')
    village_notes = list(map(parse_village, all_chunks))
    return village_notes

def map_village(village: VillageNotes) -> str:
    return village.consolidated_note()

def process_village_file(input_file_name: str, batch_size: int = 100):
    """Process a village notes file and upsert embeddings into the Pinecone index."""
    villages = read_villages(input_file_name)
    chunks = [map_village(v) for v in villages]
    embeddings = embedList(chunks)
    index = pc.Index(index_name, grpc_config=GRPCClientConfig(secure=False))
    vectors_batch = []
    for i, embedding in enumerate(embeddings):
        chunk_text = chunks[i]
        deterministic_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"village::{chunk_text}").hex
        vec = {
            'id': deterministic_id,
            'values': embedding,
            'metadata': {'chunk': chunk_text, 'type': 'village', 'village': villages[i].village}
        }
        vectors_batch.append(vec)
        if len(vectors_batch) >= batch_size:
            index.upsert(vectors=vectors_batch)
            vectors_batch = []
    if vectors_batch:
        index.upsert(vectors=vectors_batch)
    print(f"Village file processed and upserted {len(chunks)} results to Pinecone index in batches (batch_size={batch_size}).")


def main():
    parser = argparse.ArgumentParser(description="Process village notes and upsert to Pinecone index.")
    parser.add_argument("input_file", help="Path to the input file containing village notes.")
    args = parser.parse_args()

    print(f"Input file: {args.input_file}")

    create_index_if_not_exists(pc)

    process_village_file(args.input_file)

if __name__ == "__main__":
    main()
