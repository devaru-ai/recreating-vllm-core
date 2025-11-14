import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.loader import load_model_and_tokenizer, encode, decode
from engine.request import Request, RequestStatus

def test_model_loader():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    text = "Hello, world!"
    token_ids = encode(text, tokenizer)
    assert isinstance(token_ids, list)
    assert token_ids, "Encoding output is empty"
    recovered = decode(token_ids, tokenizer)
    assert isinstance(recovered, str)
    assert "Hello" in recovered

def test_request_object():
    request = Request([10, 11, 12], max_tokens=2)
    assert request.status == RequestStatus.WAITING
    request.update_status(RequestStatus.RUNNING)
    assert request.status == RequestStatus.RUNNING
    assert not request.is_completed()
    request.append_output_token(15)
    assert not request.is_completed()
    request.append_output_token(16)
    assert request.is_completed()
    request.set_block_table({0: 0, 1: 1})
    assert request.block_table[1] == 1

if __name__ == "__main__":
    print("Testing loader and tokenizer...")
    test_model_loader()
    print("Loader and tokenizer OK.")
    print("Testing Request object...")
    test_request_object()
    print("Request object OK.")
