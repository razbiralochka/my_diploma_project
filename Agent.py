import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from collections import deque
import numpy as np


loss_object = tf.keras.losses.MeanSquaredError()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


class DQN(Model):
  def __init__(self):
    super().__init__()
    self.d1 = Dense(5, activation='linear')
    self.d2 = Dense(500, activation='leaky_relu')
    self.d3 = Dense(1000, activation='linear')
    self.d4 = Dense(100, activation='leaky_relu')
    self.d5 = Dense(3, activation='linear')

  def call(self, x):

    x = tf.reshape(x, [1, 5])
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.d5(x)
    return x


class Memory():
    def __init__(self):
        self.replay_memory = deque(maxlen=15000)



class Agent():
    def __init__(self):
        self.HAL9000 = DQN()
        self.HAL9000_target = DQN()
        #self.HAL9000_target.load_weights('wheights')
        #self.HAL9000.load_weights('wheights')
        self.memory  =deque(maxlen=3000)
        self.epsilon = 1
        self.th = 0
    def select_action(self,state):


        state[0] = state[0] % (2*np.pi)
        predictions = self.HAL9000(state, training=False).numpy()
        res = np.argmax(predictions[0])

        if np.random.rand() < self.epsilon:
            res = random.randint(0,2)

        if abs(state[3]) > 0.7:
            res = round(-np.sign(state[3]))+1

        return res

    def remember(self, state, action, state_, reward, done):
            self.memory.append((state, action, state_, reward, done))
    @tf.function
    def train_step(self,state, target):
        with tf.GradientTape() as tape:
            predictions = self.HAL9000(state, training=True)
            loss = loss_object(target, predictions)
        gradients = tape.gradient(loss, self.HAL9000.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.HAL9000.trainable_variables))
        train_loss(loss)
        train_accuracy(target, predictions)

    def learn(self):
        if self.epsilon > 0.01:
            self.epsilon *= 0.995
        batch_size = 499
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        samples.append(self.memory[-1])
        for state, action, state_, reward, done in samples:

            target = self.HAL9000_target(state,training=False).numpy()
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.HAL9000_target(state_,training=False).numpy()[0])
                target[0][action] = reward + Q_future
            self.train_step(state, target)

