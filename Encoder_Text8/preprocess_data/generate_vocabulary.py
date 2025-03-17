import sentencepiece as spm

# Define the path where you want to save your model files
model_output_path = '../models/spm_vocab_text8_32k'

# Train the SentencePiece model
spm.SentencePieceTrainer.train(
    input='../data/cleaned_text8.txt',  # Your input text file
    model_prefix=model_output_path,  # Path to save the model (without the file extension)
    vocab_size=32000,  # Desired vocabulary size
    character_coverage=0.995,  # How much of the characters in the corpus to cover
    model_type='unigram'  # Model type (unigram, bpe, etc.)
)
