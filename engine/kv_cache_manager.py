import torch

class KVCacheManager:
    """
    Manages a global pool of KV-cache blocks for transformer attention.
    Each block holds storage for a fixed number of tokens.
    """
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim, device):
        """
        Pre-allocates pool of blocks for both key and value.
        All sizes follow (num_blocks, block_size, num_layers, num_heads, head_dim).
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device

        # For each layer, we keep a pool of [num_blocks, block_size, num_heads, head_dim]
        self.key_blocks = [
            torch.zeros((num_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
            for _ in range(num_layers)
        ]
        self.value_blocks = [
            torch.zeros((num_blocks, block_size, num_heads, head_dim), dtype=torch.float16, device=device)
            for _ in range(num_layers)
        ]
        self.free_block_ids = list(range(num_blocks))  # available block indices
        self.block_usage = {}  # request_id -> [list of physical block indices]

    def allocate_blocks(self, request_id, num_blocks_needed):
        """
        Allocates physical blocks for a request, returns block indices.
        """
        if len(self.free_block_ids) < num_blocks_needed:
            raise RuntimeError("Not enough free KV cache blocks available!")
        allocated = [self.free_block_ids.pop(0) for _ in range(num_blocks_needed)]
        self.block_usage[request_id] = allocated
        return allocated

    def free_blocks(self, request_id):
        """
        Frees all blocks associated with the given request.
        """
        if request_id not in self.block_usage:
            return
        for block_id in self.block_usage[request_id]:
            self.free_block_ids.append(block_id)
        del self.block_usage[request_id]

    def get_block_indices(self, request_id):
        """
        Returns list of block indices for the request.
        """
        return self.block_usage.get(request_id, [])

    def get_kv_blocks(self, block_indices, layer):
        """
        Returns a tuple (key_block, value_block) for the given indices and layer.
        Each is a tensor of shape [num_blocks, block_size, num_heads, head_dim]
        """
        key = self.key_blocks[layer][block_indices]
        value = self.value_blocks[layer][block_indices]
        return key, value

    def write_to_blocks(self, layer, block_indices, token_offset, key_tensor, value_tensor):
        """
        Writes key/value tensors into the underlying storage for a batch of tokens.
        :param layer: which transformer layer
        :param block_indices: list of physical blockIDs to write into
        :param token_offset: offset within each block to write
        :param key_tensor: shape [batch, num_heads, head_dim]
        :param value_tensor: shape [batch, num_heads, head_dim]
        """
        for i, block_id in enumerate(block_indices):
            self.key_blocks[layer][block_id, token_offset] = key_tensor[i]
            self.value_blocks[layer][block_id, token_offset] = value_tensor[i]

    def num_free_blocks(self):
        return len(self.free_block_ids)
