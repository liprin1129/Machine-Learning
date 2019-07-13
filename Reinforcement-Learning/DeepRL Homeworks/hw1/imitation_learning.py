import numpy as np
#import gym
import pickle
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from time import time # to make log

def main(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    obs = data["observations"]
    actions = data["actions"]

    # dtype conversion from float64 to float16
    obs = obs.astype(np.float16)
    #actions = np.squeeze(actions.astype(np.float16), 1)
    actions = actions.astype(np.float16).squeeze(1)

    print(obs.shape, actions.shape)
    print(type(obs[0, 0]), type(actions[0, 0]))


    inputs = Input(shape=(376,), name='observations')
    #hidden = layers.Dense(1280, activation='relu')(inputs)
    hidden = layers.Dense(500)(inputs)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Activation('relu')(hidden)

    hidden = layers.Dense(300)(hidden)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Activation('relu')(hidden)

    outputs = layers.Dense(17, activation='relu')(hidden)
    
    model = Model(inputs=inputs, outputs=outputs, name='imitation_learning')
    #model.summary()
    #keras.utils.plot_model(model, "IL-model.png", show_shapes=True)
    tensorboard = TensorBoard(log_dir='tensorboard-logs/{0}'.format(time()))

    x_train, x_test, y_train, y_test = train_test_split(obs, actions, test_size=0.2, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=500, epochs=50, validation_split=0.2, callbacks=[tensorboard])
    test_scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])


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