
import os
import pathlib

BASE_PATH = pathlib.Path(__file__).parent.parent.resolve()
CACHE_PATH = ".cache/models"

os.environ['TRANSFORMERS_CACHE'] = BASE_PATH / CACHE_PATH

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("prajjwal1/bert-tiny", torch_dtype="auto")