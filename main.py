from rlgym.rocket_league.reward_functions import CombinedReward
from rlgym.rocket_league.reward_functions.goal_reward import GoalReward
from rlgym.rocket_league.reward_functions.touch_reward import TouchReward
from rlgym_ppo.ppo import PPOLearner

from rl_state_tester.global_harvesters.callbacks import MultiCallback
from rl_state_tester.init import run
from rl_state_tester.live_testing.controls import ControlType
from rl_state_tester.live_testing.live_player import LivePlaying
from rl_state_tester.make import make
from rl_state_tester.reward_state_replayer import RewardStateReplayer
from rl_state_tester.utils.commands.commands import LivePlayingCommands

cb = CombinedReward(
    (GoalReward(), 1),
    (TouchReward(), 15)
)

callbacks = [
]


def create_env():
    return make(reward_fn=cb, renderer=None, callback=MultiCallback(callbacks))


if __name__ == "__main__":
    # env = create_env()
    #
    # agent = PPOLearner(
    #     obs_space_size=env.rlgym_env.obs_builder.get_obs_space('blue-0'),
    #     act_space_size=env.rlgym_env.action_parser.get_action_space('blue-0'),
    #     device="cpu",
    #     batch_size=100_000,
    #     mini_batch_size=10_000,
    #     n_epochs=10,
    #     continuous_var_range=(0.1, 1.0),
    #     policy_type=0,
    #     critic_layer_sizes=(256, 256, 256),
    #     policy_layer_sizes=(256, 256, 256),
    #     critic_lr=1e-5,
    #     policy_lr=1e-5,
    #     clip_range=0.2,
    #     ent_coef=0.005,
    # )
    #
    #
    # run(env, agent, with_ui=True)
    LivePlaying(
        commands=LivePlayingCommands(),
        path_to_config="TAInput.ini",
        device=ControlType.JOYSTICK,
    )

