
import os
from huggingface_hub import hf_hub_download

# Your Hugging Face repository ID
repo_id = "RaghibRizwan/minigpt-checkpoint"

# Only required if your repository is private
token = os.getenv("HF_TOKEN")

os.makedirs("checkpoints", exist_ok=True)

files = [
    "best_model.pt",
    "tokenizer.json",
]

for filename in files:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="checkpoints",
        token=token,
    )
    print(f"Downloaded {filename}: {path}")

print("Model files are ready.")
