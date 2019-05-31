import tensorflow as tf
import numpy as np
import gym
from tqdm import tqdm

from time import sleep
import sys

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

    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs = input_state, num_outputs = 10, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = action_size, activation_fn= tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
    
    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size, activation_fn= None, weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using 
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
        negative_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = output_actions)
        loss = tf.reduce_mean(negative_log_prob * montecarlo_reward_estimate)

    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Set up tensorboard
import os
if not os.path.exists('summaries'):
    os.mkdir('summaries')
if not os.path.exists(os.path.join('summaries','catpole_policy_gradient')):
    os.mkdir(os.path.join('summaries','catpole_policy_gradient'))

## Losses
tf.summary.scalar("Loss", loss)

## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)

## Summary writer
write_op = tf.summary.merge_all()

# Training
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(os.path.join('summaries','catpole_policy_gradient'), sess.graph)

    for episode in tqdm(range(max_episodes)):
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()
        #print(state)
        #env.render()
        
        while True:
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_state: state.reshape([1, 4])})
            #print(action_probability_distribution)
            
            # Sample an action
            action_sample = np.random.choice(range(action_probability_distribution.shape[1]), p = action_probability_distribution.ravel())
            """
            with tf.variable_scope("action_sample", reuse=True):
                dist = tf.contrib.distributions.Categorical(logits=action_probability_distribution)
                action_sample = dist.sample(1)
                #print('sampled action shape',action_sample.get_shape())
                action_sample = tf.squeeze(action_sample, axis=0)
            """
            #print(action_sample)
            #print()
            
            # Perfom action, a
            new_state, reward, done, info = env.step(action_sample)

            # Store s, a, r
            ## For actions because we output only one (the index) we need 2 (1 is for the action taken)
            ## We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(action_size)
            action_[action_sample] = 1

            episode_states.append(state)
            episode_actions.append(action_)
            episode_rewards.append(reward)

            if done:
                episode_rewards_sum = np.sum(episode_rewards)
                allRewards.append(episode_rewards_sum)
                total_rewards = np.sum(allRewards)
                mean_reward = total_rewards / len(allRewards)

                # Calculate discounted reward
                discounted_episode_rewards = montecarlo_reward_estimator(episode_rewards)
                #print(discounted_episode_rewards)

                # Feedforward, gradient and backpropagation
                loss_done, _ = sess.run([loss, train_opt], feed_dict={
                    input_state: np.vstack(np.array(episode_states)), 
                    output_actions: np.vstack(np.array(episode_actions)), 
                    montecarlo_reward_estimate: discounted_episode_rewards})

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={
                    input_state: np.vstack(np.array(episode_states)),
                    output_actions: np.vstack(np.array(episode_actions)),
                    montecarlo_reward_estimate: discounted_episode_rewards,
                    mean_reward_: mean_reward})
                
                summary_writer.add_summary(summary, episode)
                break

            state = new_state

"""
# Save Model
if episode % 100 == 0:
    saver.save(sess, "./models/model.ckpt")
    print("Model saved")
"""