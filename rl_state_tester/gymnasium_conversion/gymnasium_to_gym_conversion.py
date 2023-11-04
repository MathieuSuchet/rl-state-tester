import os
import warnings
import zipfile
import io
import torch as th

import gym
import gymnasium.spaces
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import json_to_data, recursive_setattr


def convert_model_to_gym(model_path: str, env, device):
    DATA_PATH = "data"

    with zipfile.ZipFile(model_path) as model_file:
        json_string = model_file.read(DATA_PATH).decode()
        data = json_to_data(json_string)

        obs_space = data['observation_space']
        action_space = data['action_space']
        if isinstance(obs_space, gymnasium.spaces.Box):
            data['observation_space'] = gym.spaces.Box(obs_space.low, obs_space.high, obs_space.shape, obs_space.dtype)
        elif isinstance(obs_space, gymnasium.spaces.Dict):
            data['observation_space'] = gym.spaces.Dict(obs_space.spaces)
        elif isinstance(obs_space, gymnasium.spaces.Tuple):
            data['observation_space'] = gym.spaces.Tuple(obs_space.spaces)
        elif isinstance(obs_space, gymnasium.spaces.Discrete):
            data['observation_space'] = gym.spaces.Discrete(obs_space.n)
        elif isinstance(obs_space, gymnasium.spaces.MultiDiscrete):
            data['observation_space'] = gym.spaces.MultiDiscrete(obs_space.nvec, obs_space.dtype)
        elif isinstance(obs_space, gymnasium.spaces.MultiBinary):
            data['observation_space'] = gym.spaces.MultiBinary(obs_space.n)

        if isinstance(action_space, gymnasium.spaces.Box):
            data['action_space'] = gym.spaces.Box(action_space.low, action_space.high, action_space.shape,
                                                  action_space.dtype)
        elif isinstance(action_space, gymnasium.spaces.Dict):
            data['action_space'] = gym.spaces.Dict(action_space.spaces)
        elif isinstance(action_space, gymnasium.spaces.Tuple):
            data['action_space'] = gym.spaces.Tuple(action_space.spaces)
        elif isinstance(action_space, gymnasium.spaces.Discrete):
            data['action_space'] = gym.spaces.Discrete(action_space.n)
        elif isinstance(action_space, gymnasium.spaces.MultiDiscrete):
            data['action_space'] = gym.spaces.MultiDiscrete(action_space.nvec, action_space.dtype)
        elif isinstance(action_space, gymnasium.spaces.MultiBinary):
            data['action_space'] = gym.spaces.MultiBinary(action_space.n)

            # noinspection PyArgumentList
        model = PPO(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        model.__dict__.update(data)
        model._setup_model()

        namelist = model_file.namelist()
        params = {}

        pth_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] == ".pth"]
        for file_path in pth_files:
            with model_file.open(file_path, mode="r") as param_file:
                # File has to be seekable, but param_file is not, so load in BytesIO first
                # fixed in python >= 3.7
                file_content = io.BytesIO()
                file_content.write(param_file.read())
                # go to start of file
                file_content.seek(0)
                # Load the parameters with the right ``map_location``.
                # Remove ".pth" ending with splitext
                th_object = th.load(file_content, map_location=device)
                # "tensors.pth" was renamed "pytorch_variables.pth" in v0.9.0, see PR #138
                if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                    # PyTorch variables (not state_dicts)
                    pytorch_variables = th_object
                else:
                    # State dicts. Store into params dictionary
                    # with same name as in .zip file (without .pth)
                    params[os.path.splitext(file_path)[0]] = th_object

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

                # put other pytorch variables back in place
            if pytorch_variables is not None:
                for name in pytorch_variables:
                    # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                    # This happens when using SAC/TQC.
                    # SAC has an entropy coefficient which can be fixed or optimized.
                    # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                    # otherwise it is initialized to `None`.
                    if pytorch_variables[name] is None:
                        continue
                    # Set the data attribute directly to avoid issue when using optimizers
                    # See https://github.com/DLR-RM/stable-baselines3/issues/391
                    recursive_setattr(model, name + ".data", pytorch_variables[name].data)

                # Sample gSDE exploration matrix, so it uses the right device
                # see issue #44
            if model.use_sde:
                model.policy.reset_noise()  # pytype: disable=attribute-error

        return model
