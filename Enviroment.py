import tensorflow as tf
import math

state = tf.Variable([0, 0, 0, 0, 0], dtype=float)
class Enviroment():
    def __init__(self):
        self.end_point = tf.constant([46, math.radians(28.6)])
        self.end_norm = tf.norm(self.end_point)

        self.acc = tf.constant(0.00351598)
        self.gain = tf.constant(0.0024262)

        self.Time = tf.Variable(0, dtype=float)

    def reset(self):
        radius = 1
        inclination = math.radians(51)
        azimuth = 0
        psi = 0
        beta = 0
        state = tf.Variable([azimuth, radius, inclination, psi, beta])

        return state, False

    #@tf.function
    def step(self, action):
        h = tf.constant(0.1)
        done = False

        az, radius, inclination, psi, beta = tf.unstack(state)

        dpsi = beta
        dbeta = tf.cast(action-1, float) * self.gain
        daz = (1 / (radius ** 1.5))
        dr = 2 * self.acc * (radius ** 1.5) * tf.cos(psi)
        di = self.acc * tf.sin(psi) * (radius ** 0.5) * tf.cos(az)

        dstate = tf.stack([daz, dr, di, dpsi, dbeta])

        state.assign_add(dstate*h)

        err_d = tf.norm(state[1:3] - self.end_point)

        cos_d = 150*tf.keras.losses.cosine_similarity(self.end_point, state[1:3])

        self.Time.assign_add(h)

        if self.Time > 250:
            done = True
        if abs(state[3]) > 1:
            done = True


        reward = -cos_d/err_d-10*di


        return state, reward, done, self.Time