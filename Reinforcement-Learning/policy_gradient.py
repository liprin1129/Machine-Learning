import tensorflow as tf
import numpy as np
import gym

# Create our environment
env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

#  Set up our hyperparameters
## ENVIRONMENT Hyperparameters
state_size = 4
action_size = env.action_space.n

## TRAINING Hyperparameters
max_episodes = 10000
learning_rate = 0.01
gamma = 0.95

def montecarlo_reward_estimator(episode_rewards):
    """
    Case 1: trajectory-based PG 

        (reward_to_go = False)

        Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        entire trajectory (regardless of which time step the Q-value should be for). 

        For this case, the policy gradient estimator is

            E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]

        where

            Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.

        Thus, you should compute

            Q_t = Ret(tau)
    """
    gamma_timestep = np.ones(len(episode_rewards)) * gamma

    gamma_timestep = np.power(gamma_timestep, range(len(episode_rewards)))
    montecarlo_reward_estimate = gamma_timestep * episode_rewards
    
    #print(montecarlo_reward_estimate)

    # Normalization
    reward_mean = np.mean(montecarlo_reward_estimate)
    
    reward_std = np.std(montecarlo_reward_estimate) +  + 1e-8
    montecarlo_reward_estimate = (montecarlo_reward_estimate - reward_mean) / reward_std
    #print(reward_mean, reward_std)

    #ratio = (montecarlo_reward_estimate - np.mean(montecarlo_reward_estimate))/(np.std(montecarlo_reward_estimate) + 1e-8)
    #montecarlo_reward_estimate = reward_mean + reward_std * ratio
    
    return montecarlo_reward_estimate

with tf.name_scope("inputs"):
    input_state = tf.placeholder(tf.float32, [None, state_size], name="states_input")
    output_actions = tf.placeholder(tf.int32, [None, action_size], name="actions_output")
    montecarlo_reward_estimate = tf.placeholder(tf.float32, [None, ], name="monte_carlo_rewards")

    