import argparse
from import_village import process_document_file, create_index_if_not_exists
from pc_client import create_pinecone_client

pc = create_pinecone_client()

def main():
    parser = argparse.ArgumentParser(description="Process vineyard notes and upsert to Pinecone index.")
    parser.add_argument("input_file", help="Path to the input file containing vineyard notes.")
    args = parser.parse_args()

    print(f"Input file: {args.input_file}")
    print(f"Data type: vineyard")

    create_index_if_not_exists(pc)

    process_document_file(args.input_file, "vineyard")

if __name__ == "__main__":
    main()
