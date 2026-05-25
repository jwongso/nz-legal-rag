"""GPU/CPU device selection utility."""

import torch


def select_device(min_vram_mb: int = 512) -> str:
    """Return 'cuda' if a GPU with at least min_vram_mb free is available, else 'cpu'."""
    if not torch.cuda.is_available():
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    return "cuda" if free_bytes >= min_vram_mb * 1024 * 1024 else "cpu"
