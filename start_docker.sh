docker run -d \
--name pinecone-local \
-e PORT=5080 \
-e PINECONE_HOST=localhost \
-p 5080-5090:5080-5090 \
--platform linux/amd64 \
ghcr.io/pinecone-io/pinecone-local:latest
