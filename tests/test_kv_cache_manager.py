import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from engine.kv_cache_manager import KVCacheManager


def test_allocation_and_free():
    """Test allocation and freeing of KV cache blocks."""
    manager = KVCacheManager(
        num_blocks=4,
        block_size=16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        device="cpu"
    )

    blocks1 = manager.allocate_blocks("req1", 2)
    assert len(blocks1) == 2
    assert manager.num_free_blocks() == 2

    blocks2 = manager.allocate_blocks("req2", 2)
    assert manager.num_free_blocks() == 0

    manager.free_blocks("req1")
    assert manager.num_free_blocks() == 2

    try:
        manager.allocate_blocks("req3", 4)
        assert False, "Over-allocation should fail"
    except RuntimeError:
        pass

    manager.free_blocks("req2")
    blocks3 = manager.allocate_blocks("req4", 2)
    assert set(blocks3) <= set(range(4))


def test_kv_block_access_and_write():
    """Test writing and reading from KV cache blocks."""
    manager = KVCacheManager(
        num_blocks=2,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        device="cpu"
    )

    req_id = "test"
    indices = manager.allocate_blocks(req_id, 2)
    assert len(indices) == 2

    # Detect dtype correctly from internal key_block tensor
    sample_block = manager.key_blocks[0]
    if isinstance(sample_block, torch.Tensor):
        example_dtype = sample_block.dtype
    elif isinstance(sample_block, list) or isinstance(sample_block, dict):
        # fallback in case it's nested per-layer data structure
        example_dtype = torch.float16
    else:
        example_dtype = torch.float16

    # Use correct dtype to match cache precision
    key = torch.ones((2, 2, 4), dtype=example_dtype)
    value = torch.zeros((2, 2, 4), dtype=example_dtype)

    # Write dummy tensors to cache
    manager.write_to_blocks(0, indices, 1, key, value)

    key_block, value_block = manager.get_kv_blocks(indices, 0)

    # Match dtype when comparing
    assert torch.allclose(key_block[:, 1].to(example_dtype), key.to(example_dtype), atol=1e-3, rtol=1e-3)
    assert torch.allclose(value_block[:, 1].to(example_dtype), value.to(example_dtype), atol=1e-3, rtol=1e-3)

    manager.free_blocks(req_id)


if __name__ == "__main__":
    print("Testing KVCacheManager allocation/free...")
    test_allocation_and_free()
    print("Allocation/free OK")

    print("Testing KV block access and write...")
    test_kv_block_access_and_write()
    print("KV block access/write OK")
