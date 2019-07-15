''' #Print all environments
from gym import envs

print(envs.registry.all())'''




''' #Check out an environment
import gym

#env = gym.make('Acrobot-v1')
#env = gym.make('Humanoid-v2')
env.reset()

for _ in range(500):
    env.render()
    env.step(env.action_space.sample())
env.close()'''




'''# Observation, reward, done, and info
import gym
#env = gym.make('MountainCarContinuous-v0')
env = gym.make('CartPole-v0')
observation = env.reset()

for t in range(1000):
    env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)

    if done:
        print("Finished after {} timesteps".format(t+1))
        break

env.close()'''

#Spaces
import gym
env = gym.make('CartPole-v0')
print(env.action_space) #[Output: ] Discrete(2)
print(env.observation_space) # [Output: ] Box(4,)
env = gym.make('MountainCarContinuous-v0')
print(env.action_space) #[Output: ] Box(1,)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space) #[Output: ] Box(2,)
print(env.observation_space.high)
print(env.observation_space.low)
