import json

with open('vector_store/chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    chunks = data['chunks']

print(f"Total chunks: {len(chunks)}\n")

# Search for Singapore and population data
for i, chunk in enumerate(chunks):
    chunk_lower = chunk.lower()
    if 'singapore' in chunk_lower or '9.7' in chunk:
        print(f"=== Chunk {i} (SINGAPORE FOUND) ===")
        # Handle encoding issues
        try:
            print(chunk.encode('ascii', 'ignore').decode('ascii')[:500])
        except:
            print(chunk[:500])
        print("...\n")

# Search for any percentage mentions with population
for i, chunk in enumerate(chunks):
    chunk_lower = chunk.lower()
    if ('population' in chunk_lower or 'prevalence' in chunk_lower or 'affected' in chunk_lower) and ('%' in chunk or 'percent' in chunk_lower):
        print(f"=== Chunk {i} (POPULATION DATA) ===")
        try:
            print(chunk.encode('ascii', 'ignore').decode('ascii')[:400])
        except:
            print(chunk[:400])
        print("...\n")