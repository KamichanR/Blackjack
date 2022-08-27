from gym.wrappers import FlattenObservation
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import suite_gym, TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common
import tensorflow as tf
import numpy as np


np.random.seed(seed=42)


# Hyperparameters
ENVIRONMENT_NAME = 'MyBlackjack-v0'
FC_LAYER_PARAMS = (100, 50)
DROPOUT_LAYER_PARAMS = (0.2, 0.2)
LEARNING_RATE = 0.01
REPLAY_BUFFER_MAX_LENGTH = 100000
SAMPLE_BATCH_SIZE = 64
NUM_EPISODE = 1000000
LOG_INTERVAL = 1000
EVAL_INTERVAL = 1000


# Environment
train_environment = TFPyEnvironment(
    suite_gym.load(
        ENVIRONMENT_NAME,
        gym_env_wrappers=[FlattenObservation],
    )
)
eval_environment = TFPyEnvironment(
    suite_gym.load(
        ENVIRONMENT_NAME,
        gym_env_wrappers=[FlattenObservation],
    )
)

# Agent
q_network = QNetwork(
    input_tensor_spec=train_environment.observation_spec(),
    action_spec=train_environment.action_spec(),
    fc_layer_params=FC_LAYER_PARAMS,
    dropout_layer_params=DROPOUT_LAYER_PARAMS,
    activation_fn=tf.keras.activations.tanh,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
train_step_counter = tf.Variable(0)
agent = DqnAgent(
    time_step_spec=train_environment.time_step_spec(),
    action_spec=train_environment.action_spec(),
    q_network=q_network,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
)
agent.initialize()

# Data Collection
replay_buffer = TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_environment.batch_size,
    max_length=REPLAY_BUFFER_MAX_LENGTH,
)
collect_driver = DynamicEpisodeDriver(
    env=train_environment,
    policy=agent.collect_policy,
    observers=[replay_buffer.add_batch],
)
collect_driver.run()
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=SAMPLE_BATCH_SIZE,
    num_steps=2,
    single_deterministic_pass=False,
).prefetch(3)
iterator = iter(dataset)


# Metrics and Evaluation
def compute_avg_return(train_environment, policy, num_episodes=100):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = train_environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = train_environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# Train
def train_one_episode(time_step):
    collect_driver.run(time_step)
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss
    iteration = agent.train_step_counter.numpy()
    return time_step, iteration, train_loss


agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

time_step = train_environment.reset()
avg_return = compute_avg_return(train_environment, agent.policy)
returns = [avg_return]

for _ in range(NUM_EPISODE):
    time_step, iteration, train_loss = train_one_episode(time_step)
    
    if iteration % LOG_INTERVAL == 0:
        print(f'iteration: {iteration}, loss: {train_loss}')
    
    if iteration % EVAL_INTERVAL == 0:
        avg_return = compute_avg_return(eval_environment, agent.policy)
        print(f'iteration: {iteration}, Average Return = {avg_return}')
        returns.append(avg_return)
