# File: inference_test.py

import torch
import torch.nn.functional as F
import sentencepiece as spm
from decoder import MyDecoder  # Ensure this file is in your PYTHONPATH or same directory
import math

def generate_text(model, init_sequence, gen_steps=10, device='cpu'):
    """
    Autoregressively generate additional tokens given an initial sequence.
    """
    model.eval()
    generated = init_sequence.copy()
    with torch.no_grad():
        for _ in range(gen_steps):
            inp = torch.tensor([generated], device=device)  # shape: [1, current_length]
            output = model(inp)  # shape: [1, current_length, vocab_size]
            next_token = output[0, -1].argmax(dim=-1).item()  # get token with highest probability
            generated.append(next_token)
    model.train()
    return generated

def main():
    # -------------------------
    # Hyperparameters & Paths
    # -------------------------
    model_save_path = './models/GPT_only_decoding_model.pth'
    sp_model_file = './models/spm_vocab_text8_32k.model'
    seq_length = 32              # Should match training config
    d_model = 64
    num_layers = 4
    d_ff = 256
    vocab_size = 32000           # Should match your SentencePiece vocab size
    gen_steps = 20               # Number of tokens to generate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Load SentencePiece Model
    # -------------------------
    sp = spm.SentencePieceProcessor(model_file=sp_model_file)

    # -------------------------
    # Instantiate & Load the Model
    # -------------------------
    model = MyDecoder(
        vocab_size=vocab_size,
        max_seq_length=seq_length,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        device=device
    ).to(device)

    # Load saved weights
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # -------------------------
    # Prompt and Inference
    # -------------------------
    prompt_text = input("Enter prompt text: ")
    # Encode prompt to token IDs using SentencePiece
    init_sequence = sp.encode(prompt_text)
    print("Encoded prompt:", init_sequence)

    # Generate text tokens
    generated_sequence = generate_text(model, init_sequence, gen_steps=gen_steps, device=device)
    
    # Decode tokens back to text
    generated_text = sp.decode(generated_sequence)
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
