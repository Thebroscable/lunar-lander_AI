from keras.models import load_model
import numpy as np
from ReplayBuffer import ReplayBuffer
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from keras import activations
from keras.callbacks import Callback
from keras import backend as k
import gc
# dqn_model.h5


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


def build_model(input_dim, n_actions, lr):
    model = Sequential([
        Dense(128, input_shape=input_dim),
        Activation(activations.relu),
        Dense(128),
        Activation(activations.relu),
        Dense(n_actions)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return model


class Agent:
    def __init__(self, input_dim, n_actions, lr, gamma,
                 epsilon_end, epsilon=1.0, epsilon_dec=0.999):
        self.n_actions = n_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end

        self.q_network = build_model(input_dim, n_actions, lr)
        self.memory = ReplayBuffer(1000000, input_dim)

    def store_data(self, state, action,
                   reward, next_state, done):
        self.memory.store_data(state, action,
                               reward, next_state,
                               done)

    def make_action(self, state):
        state = state[np.newaxis, :]

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            actions = self.q_network.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def train(self, batch_size):
        if batch_size > self.memory.mem_cntr:
            return

        states, actions, rewards, next_states, done = \
            self.memory.sample_data(batch_size)

        q_target = self.q_network.predict(states, verbose=0)
        q_next = self.q_network.predict(next_states, verbose=0)

        batch_index = np.arange(batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
            self.gamma * np.max(q_next, axis=1)*done
        self.q_network.fit(states, q_target, verbose=0,
                           callbacks=ClearMemory())

        self.epsilon = self.epsilon * self.epsilon_dec \
            if self.epsilon > self.epsilon_end \
            else self.epsilon_end

    def save_model(self, model_file='model/dqn_model.h5'):
        self.q_network.save(model_file)

    def load_model(self, model_file='model/dqn_model.h5'):
        self.q_network = load_model(model_file)
