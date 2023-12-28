import copy

from rlgym_ppo.ppo import PPOLearner
from rlgym_ppo.util import RLGymV2GymWrapper
from torch import Tensor

from rl_state_tester.utils.envs import HarvestableEnv


# def __create_action_dict(n_cars, actions):
#     indexes = [*["blue-" + str(i) for i in range(3)], *["orange-" + str(i) for i in range(3)]]
#     actions_dict = {}
#     actions_copy = actions.copy()
#     if n_cars % 2 == 0:
#         # Spawn opponents active
#         for i in range(n_cars // 2):
#             actions_dict.setdefault(indexes[i], actions_copy[i])
#             actions_dict.setdefault(indexes[i + 3], actions_copy[i + 3])
#     return actions_dict


def run(env: RLGymV2GymWrapper, agent: PPOLearner, n_steps: int = -1, agent_tick_skip: int = 8):
    obs = env.reset()
    t = 0
    current_actions = None
    while n_steps < 0 or t < n_steps:
        predicted_actions = agent.policy.get_action(obs)[0].detach().cpu().numpy()
        # predicted_actions = predicted_actions.reshape((1, *predicted_actions.shape))
        if t % agent_tick_skip == 0:
            current_actions = predicted_actions
        print(current_actions)
        obs, reward, truncated, terminal = env.step(current_actions)

        if truncated or terminal:
            obs = env.reset()

        t += 1
