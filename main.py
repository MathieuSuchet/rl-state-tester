from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.rocket_league.reward_functions.goal_reward import GoalReward
from rlgym.rocket_league.reward_functions.touch_reward import TouchReward
from rlgym_ppo.ppo import PPOLearner

from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.init import run
from rl_state_tester.live_testing.live_player import LivePlaying
from rl_state_tester.make import make
from rl_state_tester.utils.commands.commands import LivePlayingCommands

cb = CombinedReward(
    (GoalReward(), 1),
    (TouchReward(), 15)
)


def create_env():
    return make(reward_fn=cb, renderer=None, harvester=MultiCallback(
        callbacks=[
            LivePlaying(LivePlayingCommands())
        ]
    ))


if __name__ == "__main__":


    agent = PPOLearner(
        172,
        8,
        device="cpu",
        batch_size=100_000,
        mini_batch_size=10_000,
        n_epochs=10,
        continuous_var_range=(0.1, 1.0),
        policy_type=0,
        critic_layer_sizes=(256, 256, 256),
        policy_layer_sizes=(256, 256, 256),
        critic_lr=1e-5,
        policy_lr=1e-5,
        clip_range=0.2,
        ent_coef=0.005,
    )

    run(create_env(), agent)
