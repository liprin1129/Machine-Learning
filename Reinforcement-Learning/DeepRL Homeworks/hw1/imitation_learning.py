import numpy as np
import gym
import pickle
import tensorflow as tf
from tensorflow.math import reduce_mean, reduce_std, subtract, divide
from os import path
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from time import time # to make log


def load_expert(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    obs = data["observations"]
    actions = data["actions"]

    # dtype conversion from float64 to float16
    #obs = obs.astype(np.float16)
    #actions = actions.astype(np.float16).squeeze(1)

    print(obs.shape, actions.shape)
    print(type(obs[0, 0]), type(actions[0, 0]))

    return obs, actions.squeeze(1)

def train_expert(obs, actions):
    checkpoint_path = "agent_data/check-points/cp-epoch{epoch:04d}.ckpt"
    
    #checkpoint_dir = path.dirname(checkpoint_path)
    #print(checkpoint_dir)
    
    # Create checkpoint callback
    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, verbose=False, save_best_only=True, mode='min')#, save_freq=10)


    inputs = Input(shape=(376,), name='observations')
    #hidden = InputBatchNormalization()(inputs)
    hidden = layers.BatchNormalization()(inputs)
    
    #hidden = layers.Dense(1280, activation='relu')(inputs)
    hidden = layers.Dense(500)(inputs)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Activation('tanh')(hidden)

    hidden = layers.Dense(300)(hidden)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Activation('tanh')(hidden)

    outputs = layers.Dense(17)(hidden)
    #outputs = layers.Dense(17)(hidden)
    #ILRelu = lambda x: keras.activations.relu(x, alpha=0.0, max_value=0.4, threshold=-0.4)
    #outputs = layers.Activation(ILRelu)(outputs)
    #outputs = layers.Activation('tanh')(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='imitation_learning')

    model.save("agent_data/model/IL-model.h5") # Save model

    tensorboard = TensorBoard(
        #log_dir='tensorboard-logs/{0}'.format(time()),
        log_dir='tensorboard-logs/train-log',
        histogram_freq = 1,
        write_graph = True, 
        write_grads = True, 
        write_images = 1)

    x_train, x_test, y_train, y_test = train_test_split(obs, actions, test_size=0.2, random_state=42)
    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, batch_size=500, epochs=500, validation_split=0.2, callbacks=[tensorboard, cp_callback])
    #history = model.fit(x_train, y_train, batch_size=500, epochs=500, validation_split=0.2, callbacks=[tensorboard])

    test_scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

def action_inference(obs):
    latest = tf.train.latest_checkpoint("agent_data/check-points/")
    print(latest)

    model = keras.models.load_model("agent_data/model/IL-model.h5")
    #model.summary()
    model.load_weights(latest)

    action = model.predict(np.vstack([obs, obs]), batch_size=1, verbose=1)
    return action[0]

class InputBatchNormalization(layers.Layer):
    mean = 0
    std = 0
    temp = 0

    def __init__(self):
        super(InputBatchNormalization, self).__init__()
        InputBatchNormalization.mean
        InputBatchNormalization.std
        InputBatchNormalization.temp += 1

    def call(self, inputs):
        mean = reduce_mean(inputs, axis=0)
        std = reduce_std(inputs, axis=0)+1e-6

        InputBatchNormalization.temp += 1
        InputBatchNormalization.mean += mean
        InputBatchNormalization.std += std

        inputs = divide(subtract(inputs, mean), std)

        return inputs.squeeze(0)


if __name__ == "__main__":
    obs, actions = load_expert("./expert_data/Humanoid-v2.pkl")

    #train_expert(obs, actions)

    env = gym.make('Humanoid-v2')
    observation = env.reset()
    #print(np.shape(np.vstack([observation, observation])))
    
    model = keras.models.load_model("agent_data/model/IL-model.h5")
    latest = tf.train.latest_checkpoint("agent_data/check-points/trained-check-points/")
    #print("\n\n", latest)
    model.load_weights(latest)
    
    for _ in range(300):
        for i in range(100):
            
            action = model.predict(np.vstack([observation, observation]), batch_size=1, verbose=0)
            #print(np.shape(action[0]))
            #print(env.action_space)
            
            #observation, reward, done, info = env.step(action[0])
            observation, reward, done, info = env.step(env.action_space.sample())
            env.render()
            #print("iteration: {0} ".format(i), np.shape(observation))
            #print(reward, done, info)

            if done:
                print("Finished after {} timesteps".format(i+1))
                #break
                observation = env.reset()


    env.close()
    