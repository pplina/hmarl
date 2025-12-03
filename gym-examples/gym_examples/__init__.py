from gymnasium.envs.registration import register

register(
     id="gym_examples/GridWorld-v0",
     entry_point="gym_examples.envs:GridWorldEnv",
     max_episode_steps=300,
)

register(
     id="gym_examples/CERERE-v0",
     entry_point="gym_examples.envs:CerereNet",
     max_episode_steps=300,
)