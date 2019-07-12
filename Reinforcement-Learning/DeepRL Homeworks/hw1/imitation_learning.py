import numpy as np
#import gym
import pickle
import tensorflow as tf

def main(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    obs = data["observations"]
    actions = data["actions"]

    # dtype conversion from float64 to float16
    obs = obs.astype(np.float16)
    actions = actions.astype(np.float16)

    print(obs.shape, actions.shape)
    print(type(obs[0, 0]), type(actions[0, 0, 0]))

    features = tf.placeholder(tf.float16, shape=[-1, 376], name="features:observation")
    labels = tf.placeholder(tf.float16, shape=[-1, 1, 17], name="labels:actions")



if __name__ == "__main__":
    main("./expert_data/Humanoid-v2.pkl")

    """
    import gym
    env = gym.make('Humanoid-v2')
    env.reset()
    action = np.zeros([1, 17])
    
    while True:
        env.step(0)
        env.render()
    """