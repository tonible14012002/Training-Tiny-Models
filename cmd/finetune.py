
import os
import pathlib

BASE_PATH = pathlib.Path(__file__).parent.parent.resolve().__str__()
CACHE_PATH = ".cache/models"

os.environ['TRANSFORMERS_CACHE'] = BASE_PATH + "/" + CACHE_PATH
print(BASE_PATH)

# Load model directly
# from transformers import pipelines
from transformers.pipelines import pipeline

model = pipeline("text-classification", model="prajjwal1/bert-tiny", torch_dtype="auto")

print(model("Hello, my dog is cute"))
