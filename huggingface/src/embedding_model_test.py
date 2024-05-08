from sentence_transformers import SentenceTransformer, models, LoggingHandler, losses, util

# Load Embedding Model
embedding_model = models.Transformer(
    model_name_or_path='./klue-sroberta-base-continue-learning-by-mnr', 
    max_seq_length=512,
    do_lower_case=True
)

# Only use Mean Pooling -> Pooling all token embedding vectors of sentence.
pooling_model = models.Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[embedding_model, pooling_model])


sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
print(embeddings)