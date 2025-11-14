import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id, device=None):
    """
    Loads model and tokenizer from Hugging Face, moves model to device.
    Returns (model, tokenizer, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

def encode(text, tokenizer):
    """
    Tokenizes given text and returns a list of token ids.
    """
    return tokenizer.encode(text, return_tensors="pt")[0].tolist()

def decode(token_ids, tokenizer):
    """
    Decodes a list of token ids to a string (removes special tokens).
    """
    return tokenizer.decode(token_ids, skip_special_tokens=True)
