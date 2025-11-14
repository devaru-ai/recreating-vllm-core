import torch
from collections import deque
import logging
import time


class EngineCore:
    """
    Core engine for LLM inference (continuous batching, paged attention).
    Handles request queuing, batching, and kv-cache management.
    """

    def __init__(self, model, tokenizer, kv_cache_manager, max_batch_size=8, device="cpu", eos_token_id=None):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache_manager
        self.device = device
        self.max_batch_size = max_batch_size
        self.eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
        self.waiting_queue = deque()
        self.running_queue = deque()
        logging.debug(f"EngineCore initialized with max_batch_size={max_batch_size}, device={device}")

    def add_request(self, request):
        """Add an inference request to the waiting queue."""
        self.waiting_queue.append(request)
        logging.debug(f"Added request {request.request_id[:8]} to waiting_queue (len={len(self.waiting_queue)})")

    def _allocate_kv_for(self, request):
        """Allocate KV cache memory for a request."""
        prompt_len = len(request.prompt_token_ids)
        num_blocks = (prompt_len + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        logging.debug(f"Allocating KV blocks for {request.request_id[:8]}: num_blocks={num_blocks}")
        block_indices = self.kv_cache.allocate_blocks(request.request_id, num_blocks)
        block_table = {i: block_indices[i // self.kv_cache.block_size] for i in range(prompt_len)}
        request.set_block_table(block_table)

    def _free_kv_for(self, request):
        """Free KV cache memory for completed request."""
        logging.debug(f"Freeing KV blocks for {request.request_id[:8]}")
        self.kv_cache.free_blocks(request.request_id)

    def _prep_batch(self, requests):
        """Prepare batched token, attention, and position tensors."""
        batch_lens = [len(r.prompt_token_ids) + len(r.output_token_ids) for r in requests]
        max_len = max(batch_lens)
        batch_tokens, batch_attention_mask, batch_position_ids = [], [], []

        for r, seq_len in zip(requests, batch_lens):
            tokens = r.prompt_token_ids + r.output_token_ids
            pad = [self.tokenizer.pad_token_id] * (max_len - seq_len)
            attention_mask = [1] * seq_len + [0] * len(pad)
            pos_ids = list(range(seq_len)) + [0] * len(pad)
            batch_tokens.append(torch.tensor(tokens + pad, dtype=torch.long, device=self.device))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long, device=self.device))
            batch_position_ids.append(torch.tensor(pos_ids, dtype=torch.long, device=self.device))

        tokens = torch.stack(batch_tokens, dim=0)
        attention_mask = torch.stack(batch_attention_mask, dim=0)
        position_ids = torch.stack(batch_position_ids, dim=0)
        logging.debug(f"Prepared batch: tokens={tokens.shape}, mask={attention_mask.shape}")
        return tokens, attention_mask, position_ids

    def run(self, max_steps=64, print_fn=print):
        """Run batched inference loop with forced finalization for all requests."""
        if not self.waiting_queue and not self.running_queue:
            logging.warning("EngineCore.run() called with empty queues.")
            return

        finished, steps = 0, 0
        logging.debug("EngineCore.run started")
        start_time = time.time()

        while self.waiting_queue or self.running_queue:
            # Move requests into running queue up to batch limit
            while len(self.running_queue) < self.max_batch_size and self.waiting_queue:
                req = self.waiting_queue.popleft()
                self._allocate_kv_for(req)
                req.update_status("running")
                self.running_queue.append(req)
                logging.debug(f"Moved request {req.request_id[:8]} to running_queue")

            # Filter completed
            self.running_queue = deque([r for r in self.running_queue if not r.is_completed()])
            if not self.running_queue:
                break

            tokens, attn_mask, pos_ids = self._prep_batch(self.running_queue)
            with torch.no_grad():
                out = self.model(input_ids=tokens, attention_mask=attn_mask, position_ids=pos_ids)
                logits = out.logits[:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1).tolist()
            logging.debug(f"Step {steps}: Next tokens {next_tokens}")

            for r, tok in zip(self.running_queue, next_tokens):
                r.append_output_token(tok)
                decoded = self.tokenizer.decode(r.output_token_ids, skip_special_tokens=True)
                print_fn(f"[{r.request_id[:8]}] >> {decoded}")
                if tok == self.eos_token_id or r.is_completed():
                    r.update_status("completed")
                    self._free_kv_for(r)
                    finished += 1
                    logging.debug(f"Request {r.request_id[:8]} completed at step {steps}")

            steps += 1
            if steps >= max_steps:
                print_fn("Aborted after max_steps limit reached.")
                logging.warning("Max step limit reached while decoding.")
                break

        duration = time.time() - start_time
        logging.debug(f"EngineCore.run completed after {steps} steps ({duration:.2f}s total). Finished={finished}")

        # --- Forced finalization for uncompleted requests ---
        if self.running_queue or self.waiting_queue:
            logging.debug(f"Forcing completion for {len(self.running_queue) + len(self.waiting_queue)} pending requests")
            for r in list(self.running_queue) + list(self.waiting_queue):
                if not r.is_completed():
                    r.update_status("terminated")
                    print_fn(f"[{r.request_id[:8]}] >> [FORCED E2E END]")
                    self._free_kv_for(r)
            self.running_queue.clear()
            self.waiting_queue.clear()
        logging.debug("EngineCore.run finalized all outstanding requests.")
