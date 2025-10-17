import requests
import os

url = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
folder = "RAG files"
os.makedirs(folder, exist_ok=True)
file_path = os.path.join(folder, "war_and_peace.txt")

if not os.path.exists(file_path):
    print("Downloading War and Peace in chunks...")
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "w", encoding="utf-8") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk.decode('utf-8', errors='ignore'))
        print(f"Download complete! Saved to {file_path}")
    except requests.RequestException as e:
        print("Error downloading file:", e)
else:
    print(f"File already exists at {file_path}")
