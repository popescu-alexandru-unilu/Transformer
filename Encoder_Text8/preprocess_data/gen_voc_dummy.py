import sentencepiece as spm

with open('./data/dummy.txt', 'w') as f:
    f.write("This is a test text file.")

# Run the SentencePiece training again with dummy.txt
spm.SentencePieceTrainer.train(
    input='./data/dummy.txt',
    model_prefix='./models/spm_vocab_dummy',
    vocab_size=15,
    character_coverage=0.995,
    model_type='unigram'
)
