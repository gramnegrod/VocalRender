import argparse
import os
from transformers import LlamaTokenizerFast

def setup_svs_tokenizer(tokenizer_path, save_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
    
    # 1. Define SVS Tokens
    # Pitch tokens: <P_0> to <P_127>
    pitch_tokens = [f"<P_{i}>" for i in range(128)]
    
    # Note duration tokens
    base_note_tokens = ["<NOTE_1>", "<NOTE_2>", "<NOTE_4>", "<NOTE_8>", "<NOTE_16>", "<NOTE_32>"]
    dotted_note_tokens = ["<NOTE_DOT_1>", "<NOTE_DOT_2>", "<NOTE_DOT_4>", "<NOTE_DOT_8>", "<NOTE_DOT_16>", "<NOTE_DOT_32>"]
    note_tokens = base_note_tokens + dotted_note_tokens
    
    # BPM tokens: <BPM_0> to <BPM_255>
    bpm_tokens = [f"<BPM_{i}>" for i in range(256)]
    
    new_tokens = pitch_tokens + note_tokens + bpm_tokens
    
    # 2. Add tokens
    print(f"Adding {len(new_tokens)} new tokens...")
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Successfully added {num_added} tokens.")
    
    # 3. Save
    print(f"Saving new tokenizer to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add SVS tokens to LlamaTokenizer")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to original tokenizer")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save new tokenizer")
    
    args = parser.parse_args()
    
    setup_svs_tokenizer(args.tokenizer_path, args.save_path)
