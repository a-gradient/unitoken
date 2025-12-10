# %%
from pathlib import Path
import requests
def download_url(url, filename):
  response = requests.get(url, stream=True)
  response.raise_for_status()
  with open(filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)

cache_dir = Path(__file__).parent / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
filename = "TinyStories_all_data_zh_1M.tar.gz"
if not (cache_dir/filename).exists():
  url = "https://huggingface.co/datasets/52AI/TinyStoriesZh/resolve/main/TinyStories_all_data_zh_1M.tar.gz"
  download_url(url, filename=cache_dir/filename)

# %%
import tarfile
name = filename.removesuffix(".tar.gz")
target_folder = cache_dir/name
if not target_folder.exists():
  target_folder.mkdir(parents=True, exist_ok=True)
  with tarfile.open(cache_dir/filename, "r:gz") as tar:
    tar.extractall(path=cache_dir/name)

# %%
import json
result: list[str] = []
for file in sorted(target_folder.glob("*.json")):
  with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)
    assert isinstance(data, list)
    print(f"{file.name}: {len(data)} records")
    result.extend([i['story'] for i in data])

# %%
import random
random.seed(42)
random.shuffle(result)
if not (cache_dir / f"{name}-train.txt").exists() or True:
  split1 = int(0.98 * len(result))
  train_data = result[:split1]
  test_data = result[split1:]
  with open(cache_dir / f"{name}-train.txt", "w", encoding="utf-8") as f:
    for item in train_data:
      f.write(item + "\n<|endoftext|>\n")

  with open(cache_dir / f"{name}-test.txt", "w", encoding="utf-8") as f:
    for item in test_data:
      f.write(item + "\n<|endoftext|>\n")

  with open(cache_dir / f"{name}-sample.txt", "w", encoding="utf-8") as f:
    for item in result[10000::97][:7000]:
      f.write(item + "\n<|endoftext|>\n")

# %%
from pathlib import Path
import numpy as np
name = "TinyStories_all_data_zh_1M"
out_dir = Path(__file__).parent / "out"
data = None
if (out_dir/f"idxs.{name}-sample.npy").exists():
  data = np.load(out_dir / f"idxs.{name}-sample.npy")
  print(data.shape)

# %%
