import uuid
from enum import Enum

class RequestStatus(Enum):
    WAITING = 'waiting'
    RUNNING = 'running'
    COMPLETED = 'completed'

class Request:
    """
    Represents a complete inference request, ready for batched scheduling.
    """
    def __init__(self, prompt_token_ids, max_tokens=32):
        self.request_id = str(uuid.uuid4())
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids = []
        self.status = RequestStatus.WAITING
        self.block_table = {}  # logical idx -> physical KV-cache block index
        self.max_tokens = max_tokens
        self.completed = False

    def append_output_token(self, token_id):
        self.output_token_ids.append(token_id)
        if len(self.output_token_ids) >= self.max_tokens:
            self.status = RequestStatus.COMPLETED
            self.completed = True

    def is_completed(self):
        return self.status == RequestStatus.COMPLETED or self.completed

    def update_status(self, status):
        self.status = status

    def set_block_table(self, block_table):
        self.block_table = block_table

    def __repr__(self):
        return (f"<Request id={self.request_id[:8]}, "
                f"status={self.status.value}, "
                f"input_len={len(self.prompt_token_ids)}, "
                f"output_len={len(self.output_token_ids)}>")
