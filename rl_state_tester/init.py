from rlgym_ppo.ppo import PPOLearner
from rlgym_ppo.util import RLGymV2GymWrapper

from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.ui.ui_handling import UIHandler
from rl_state_tester.utils.envs import HarvestableEnv


def run(env: RLGymV2GymWrapper, agent: PPOLearner, n_steps: int = -1, agent_tick_skip: int = 8, with_ui: bool = False):
    """
    Runs a classic rlgym environment process (reset, step until terminal, loop)
    :param env: The environment
    :param agent: The agent that will play
    :param n_steps: Number of steps to play (-1 if unlimited, -1 by default)
    :param agent_tick_skip: Agent tick skip (8 by default)
    :param with_ui: True if you want to link the ui (see https://github.com/MrCrystAll/state-tester-ui, disabled by default)
    """

    if with_ui:
        callbacks = []

        if isinstance(env.rlgym_env, HarvestableEnv):
            if isinstance(env.rlgym_env.callback, MultiCallback):
                callbacks = env.rlgym_env.callback.callbacks
            else:
                callbacks = [env.rlgym_env.callback]

        handler = UIHandler(callbacks)
        handler.serve()

    obs = env.reset()
    t = 0
    current_actions = None
    while n_steps < 0 or t < n_steps:
        predicted_actions = agent.policy.get_action(obs)[0].detach().cpu().numpy()
        predicted_actions = predicted_actions.reshape((*predicted_actions.shape, 1))
        if t % agent_tick_skip == 0:
            current_actions = predicted_actions
        obs, _, truncated, terminal, _ = env.step(current_actions)

        if truncated or terminal:
            obs = env.reset()

        t += 1
