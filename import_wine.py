import argparse
from pinecone.grpc import GRPCClientConfig
from pinecone import ServerlessSpec
import uuid
from wine_guide import index_name, embedList, ProducerNotes, WineNote
from pc_client import create_pinecone_client
import re
import sys

# Allowed sources. Update to add more in future
ALLOWED_SOURCES = ["New French Wine", "Bourgogne Aujourd'hui", "My Notes", "Burgundy Direct", "Other", "Clive Coates Favourite Burgundies", "Bill Nanson Burgundy"]

pc = create_pinecone_client()

def create_index():
    if not pc.has_index(name=index_name):
        print('Creating index...')
        pc.create_index(
            name=index_name,
            dimension=3072,  # dimensionality of text-embedding-large-003
            metric='dotproduct',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        print('Index already exists.')

def strip_and_fix_chars(s: str) -> str:
    s = s.strip()
    s = s.replace('–', '-')  # en-dash to hyphen
    s = s.replace('\u2019', "'")  # right single quotation mark to apostrophe
    return s

def parse_domain(chunk: str, source: str) -> ProducerNotes:
    chunk = strip_and_fix_chars(chunk)

    lines = chunk.splitlines()
    producer_name = normalize_producer(get_text_before_comma(lines[0]))

    wines = []
    producer_notes = []
    for line in lines[1:]:
        # print(f"Parsing line: {line}")
        if line.endswith('.'):
            line = line[:-1].rstrip()   # remove trailing '.'

        if line.startswith('- '):
            line = line[2:].rstrip()
            producer_notes.append(line)
        
        elif line.startswith('@ '):
            line = line[2:].rstrip()
            wines.append(WineNote(note=f"{producer_name} {line}", source=source))
        else:
            raise ValueError(f"Unrecognized line format: {line}")

    # TODO model producer village?
    dn = ProducerNotes(producer=producer_name, producer_notes=producer_notes, wines=wines, raw=chunk)
    # print(f"Parsed domaine: {dn}")
    return dn


def read_domaines(input_file_name: str) -> tuple[list[ProducerNotes], str]:
    """Read a text file and return cleaned chunks of text.

    Args:
        input_file_name (str): Path to the input text file to read.

    Returns:
        List[str]: A list of information for each domaine.

    Notes:
        - The file is split on double newlines ("\n\n"). This is because my wine notes
            are formatted with double newlines between wineries. The idea is to keep
            all the notes for each winery together in one chunk.
    """

    with open(input_file_name, 'r') as file:
        file_content = file.read()

    # All chunks separated by double newlines. First one should be the source line.
    all_chunks = file_content.split('\n\n')

    file_source = extract_and_validate_source(all_chunks[0])

    domaine_notes = list(map(lambda chunk: parse_domain(chunk, file_source), all_chunks[1:]))
    # print(f"Domaine notes: {domaine_notes}")
    return domaine_notes, file_source

def map_domaine(domaine: ProducerNotes) -> list[str]:
    chunks = []

    chunks.append(domaine.consolidated_note())
    chunks.extend([wine.note for wine in domaine.wines])

    return chunks

def process_input_file(input_file_name: str, batch_size: int = 100):
    """Process an input file and upsert embeddings into the Pinecone index in batches.

    Args:
        input_file_name (str): Path to the input text file containing wine notes.
        batch_size (int): Number of vectors to upsert in a single API call. Defaults to 100.
    """

    domaines, file_source = read_domaines(input_file_name)

    chunks = [item for domaine in domaines for item in map_domaine(domaine)]

    # print(f"Chunks: {chunks}")

    # TODO may eventually need to do this in batches if files get too big
    embeddings = embedList(chunks)

    index = pc.Index(index_name, grpc_config=GRPCClientConfig(secure=False))

    # Build vector dicts and upsert in batches to reduce API calls and improve throughput
    vectors_batch = []
    for i, embedding in enumerate(embeddings):
        # Generate a deterministic UUIDv5 based on source + chunk text so the same
        # chunk from the same source will have the same id across runs.
        # This avoids collisions when re-importing the same file.
        chunk_text = chunks[i]
        deterministic_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_source}::{chunk_text}").hex

        vec = {
            'id': deterministic_id,
            'values': embedding,
            'metadata': {'chunk': chunk_text, 'source': file_source}
        }
        vectors_batch.append(vec)

        # When batch is full, send it
        if len(vectors_batch) >= batch_size:
            index.upsert(vectors=vectors_batch)
            vectors_batch = []

    # Upsert any remaining vectors
    if vectors_batch:
        index.upsert(vectors=vectors_batch)

    print(f"Input file processed and upserted {len(chunks)} results to Pinecone index in batches (batch_size={batch_size}).")

def extract_and_validate_source(line: str) -> str:
    """Extract source from a line and validate it's in ALLOWED_SOURCES.
    
    Args:
        line (str): The line to parse for a source header.
        
    Returns:
        str: The validated source if found and allowed.
        None: If no source header is found.
        
    Raises:
        ValueError: If source is found but not in ALLOWED_SOURCES.
    """
    m = re.match(r"^Source\s*:\s*(.+)$", line, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Source was not found"
        )
    
    source = m.group(1).strip()
    
    if source not in ALLOWED_SOURCES:
        allowed = ", ".join(ALLOWED_SOURCES)
        raise ValueError(
            f"Source '{source}' is not allowed. Allowed values: {allowed}"
        )
    
    return source

def normalize_producer(producer: str) -> str:
    p = producer.title()
    if p.startswith("D."):
        p = "Domaine " + p[2:].strip()
    
    return p

def get_text_before_comma(text: str) -> str:
    """Extract characters before the first comma from a string.
    
    Args:
        text (str): The input string to process.
        
    Returns:
        str: The substring before the first comma, or the entire string if no comma is found.
    """
    return text.split(',')[0]

def main():
    parser = argparse.ArgumentParser(description="Process wine notes and upsert to Pinecone index.")
    parser.add_argument("input_file", help="Path to the input file containing wine notes.")
    args = parser.parse_args()

    print(f"Input file: {args.input_file}")
    create_index()
    process_input_file(args.input_file)

if __name__ == "__main__":
    main()
