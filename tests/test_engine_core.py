import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.kv_cache_manager import KVCacheManager
from engine.request import Request
from engine.engine_core import EngineCore
import torch

class DummyModel:
    def __init__(self, vocab_size=10, eos_id=1):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_id
    def eval(self): pass
    def to(self, device): return self
    def __call__(self, input_ids, attention_mask, position_ids):
        batch = input_ids.shape[0]
        logits = torch.zeros((batch, input_ids.shape[1], self.vocab_size))
        logits[..., self.eos_token_id] = 10  # Always select EOS
        class Out: pass
        out = Out(); out.logits = logits
        return out
class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def decode(self, ids, skip_special_tokens=True): return " ".join(str(i) for i in ids)

def test_engine_core():
    kv = KVCacheManager(
        num_blocks=8, block_size=4, num_layers=1, num_heads=2, head_dim=2, device="cpu"
    )
    engine = EngineCore(DummyModel(), DummyTokenizer(), kv, max_batch_size=3)
    reqs = [Request([2], max_tokens=2), Request([5], max_tokens=3)]
    for r in reqs: engine.add_request(r)
    outputs = []
    engine.run(print_fn=outputs.append)
    assert all(r.is_completed() for r in reqs)
    print("EngineCore test passed.")

if __name__ == "__main__":
    test_engine_core()
