"""Run-level helpers: seeded task subsets and per-cycle checkpoints.

Kept free of model and dataset imports so they can be unit-tested on their own.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fields of a model result that must match for a checkpoint to be resumed
CHECKPOINT_IDENTITY_KEYS = ("model", "max_cycles", "summary_language", "max_tasks", "subset_seed")


def select_task_subset(prompts: List[Dict[str, Any]], max_tasks: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """Picks a seeded random subset of tasks, keeping the dataset order.

    The same (max_tasks, seed) always selects the same tasks, so a baseline and
    an extension run with the same settings evaluate identical task sets.

    Args:
        prompts: Dataset prompts, each with a "task_id".
        max_tasks: Number of tasks to keep; None (or >= the dataset size) keeps all.
        seed: Seed of the random selection.

    Returns:
        The selected prompts, in their original order.
    """
    if max_tasks is None or max_tasks >= len(prompts):
        return list(prompts)
    chosen = set(random.Random(seed).sample([p["task_id"] for p in prompts], max_tasks))
    return [p for p in prompts if p["task_id"] in chosen]


def save_checkpoint(path: Path, model_result: Dict[str, Any], completed_cycle: int,
                    next_prompts: List[Dict[str, Any]]) -> None:
    """Writes the run state after a completed cycle (atomically, so a crash never leaves half a file)."""
    state = {
        "completed_cycle": completed_cycle,
        "next_prompts": next_prompts,
        "model_result": model_result,
    }
    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, default=str)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, fresh_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Loads a checkpoint written by save_checkpoint, if there is one.

    Args:
        path: Checkpoint file.
        fresh_result: The model result a new run would start from; used to make
            sure the checkpoint belongs to the same model, loop and task set.

    Returns:
        The checkpoint state, or None if the file does not exist.

    Raises:
        ValueError: If the checkpoint was written by a different configuration.
    """
    if not Path(path).exists():
        return None
    with open(path) as f:
        state = json.load(f)

    saved = state["model_result"]
    for key in CHECKPOINT_IDENTITY_KEYS:
        if saved.get(key) != fresh_result.get(key):
            raise ValueError(
                f"Checkpoint {path} has {key}={saved.get(key)!r}, but this run has {fresh_result.get(key)!r}"
            )
    saved_ids = [r["task_id"] for r in saved["prompt_results"]]
    fresh_ids = [r["task_id"] for r in fresh_result["prompt_results"]]
    if saved_ids != fresh_ids:
        raise ValueError(f"Checkpoint {path} covers a different task set than this run")
    return state
