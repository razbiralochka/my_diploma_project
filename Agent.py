import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

loss_object = tf.keras.losses.MeanSquaredError()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


class DQN(Model):
  def __init__(self):
    super().__init__()
    self.d1 = Dense(5, activation='linear')
    self.d2 = Dense(360, activation='relu')
    self.d3 = Dense(500, activation='relu')
    self.d4 = Dense(100, activation='linear')
    self.d5 = Dense(3)

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.d5(x)
    return x


class Memory():
    def __init__(self):
        self.replay_memory = list()
    def memorize(self, state, action, reward, next_state,done):
        self.replay_memory.append((state, action, reward, next_state))

    def generate_batch(self):
        if len(self.replay_memory) <= 200:
            mini_batch = self.replay_memory
        else:
            mini_batch = random.sample(self.replay_memory, 200)
        return mini_batch

class Agent():
    def __init__(self):
        self.HAL9000 = DQN()
        self.memory  = Memory()
    @tf.function
    def select_action(self,state):
        predictions = self.HAL9000(tf.reshape(state,[1,5]), training=False)
        res = tf.argmax(predictions[0])
        res = tf.cast(res,tf.float32)
        return res
    def remember(self,state, action, reward, next_state,done):
        self.memory.memorize(state, action, reward, next_state,done)
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
        train_loss.reset_states()
        train_accuracy.reset_states()
        mini_batch=self.generate_batch()
        N = len(mini_batch)
        for i,line in enumerate(mini_batch):
            state = np.array([line[0]])
            action = line[1]
            reward = line[2]
            next_state = np.array([line[3]])
            if i == N-1:
                target = reward
            else:
                target = reward + 0.99*np.amax(self.HAL9000(next_state)[0])
            target_f = self.HAL9000(state).numpy()
            target_f[0][action] = target
            self.train_step(state, target_f)
    def clear_memory(self):
        self.replay_memory.clear()