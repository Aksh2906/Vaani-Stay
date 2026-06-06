import os
from glob import glob
from dotenv import load_dotenv
from hindsight import HindsightClient

load_dotenv()

HINDSIGHT_API_KEY = os.environ.get("HINDSIGHT_API_KEY")

if not HINDSIGHT_API_KEY:
    print("Error: HINDSIGHT_API_KEY not set in .env")
    exit(1)

# Initialize Hindsight Client pointing to Vectorize Cloud
client = HindsightClient(api_key=HINDSIGHT_API_KEY, base_url="https://api.hindsight.vectorize.io")
bank_id = "vaani_knowledge_base"

def ingest_directory(directory: str):
    print(f"\nScanning {directory} for documents...")
    files = glob(os.path.join(directory, "**/*.txt"), recursive=True)
    
    if not files:
        print(f"No documents found in {directory}.")
        return

    for file_path in files:
        print(f"Ingesting {file_path} to Hindsight Cloud...")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            try:
                # Retain the document content in Hindsight Cloud
                # Passing the filename as context helps Hindsight associate metadata
                response = client.retain(
                    bank_id=bank_id, 
                    content=content,
                    context=f"Source Document: {os.path.basename(file_path)}"
                )
                print(f"  -> Successfully retained!")
            except Exception as e:
                print(f"  -> Failed to retain {file_path}: {e}")

if __name__ == "__main__":
    print(f"Starting ingestion to Hindsight Cloud (Bank: {bank_id})")
    ingest_directory("./data")
    print("\nIngestion complete!")
