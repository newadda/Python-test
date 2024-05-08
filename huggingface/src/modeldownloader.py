from huggingface_hub import hf_hub_download
import joblib
from sentence_transformers import SentenceTransformer




REPO_ID = "bespin-global/klue-sroberta-base-continue-learning-by-mnr"
FILENAME = "sklearn_model.joblib"


model = SentenceTransformer(REPO_ID)
model.save('./test')



'''
model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)
'''