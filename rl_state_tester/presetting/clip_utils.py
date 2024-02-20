import datetime
import pickle
from typing import NamedTuple, Sequence, List

import numpy as np
from rlgym.utils.gamestates import GameState

DEFAULT_CLIP_PATH = "clips"
DEFAULT_CLIP_LEGEND_PATH = "clips.txt"
DEFAULT_STEPS_SAVED = 100


class Clip(NamedTuple):
    starting_state: GameState
    actions: Sequence[np.array]
    name: str


def register_clips(clips: List[Clip], state, actions, clip_file_path, legend_clip_file_path):
    name = "Clip_" + datetime.datetime.now().strftime("%Y%d%m%H%M%S%f")
    clips.append(Clip(
        starting_state=state,
        actions=actions,
        name=name
    ))

    save_clips(clip_file_path, legend_clip_file_path, clips)
    print(f"{name} saved successfully!")


def save_clips(path, legend_path, clips):
    try:
        print("Saving clips...")
        with open(path, "wb") as f:
            f.write(pickle.dumps(clips))
        with open(legend_path, "w") as fc:
            for i, c in enumerate(clips):
                fc.write(f"{i} - {c.name}\n")
    except Exception as e:
        print("Error while saving clips:", e)


def load_clips(path):
    try:
        print("Loading clips...")
        with open(path, "rb") as f:
            pickle_bytes = f.read()
            return pickle.loads(pickle_bytes)
    except Exception as e:
        print("Couldn't load clips:", e)
