import numpy as np
from rlgym.rocket_league.reward_functions.goal_reward import GoalReward
from rlgym.rocket_league.reward_functions.touch_reward import TouchReward
from rlgym_ppo import Learner

from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.reward_state_replayer import RewardStateReplayer
from rl_state_tester.make import make
from rl_state_tester.ui.ui_handling import UIHandler
from rl_state_tester.utils.rewards.common_rewards import SplitCombinedReward

cb = SplitCombinedReward(
    (GoalReward(), 1),
    (TouchReward(), 15)
)

callbacks = [
    RewardStateReplayer(rendered=False, combined_reward=cb)
]

def create_env():
    return make(reward_fn=cb, renderer=None, harvester=MultiCallback(callbacks))


if __name__ == "__main__":

    env = create_env()

    agent = Learner(
        env_create_function=create_env,
        ppo_batch_size=100,
        ppo_minibatch_size=10,

        timestep_limit=100,
        ts_per_iteration=100,

        exp_buffer_size=100,
        device="cpu",
        gae_lambda=0.99,
        n_proc=1,
        critic_lr=1e-5,
        policy_lr=1e-5,

        critic_layer_sizes=(256, 256, 256),
        policy_layer_sizes=(256, 256, 256),
        load_wandb=False,
    )

    handler = UIHandler(callbacks)
    handler.serve()

    obs = env.reset()
    for i in range(100):
        actions, _ = agent.agent.policy.get_action(obs)
        actions = actions.numpy().astype(np.float32)
        actions = actions.reshape((*actions.shape, 1))
        obs, reward, terminated, truncated, _ = env.step(actions)

        if terminated or truncated:
            obs = env.reset()

    env.close()
