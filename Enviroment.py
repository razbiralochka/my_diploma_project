import tensorflow as tf
import math


class Enviroment():
    def __init__(self):
        self.end_point = tf.constant([46, math.radians(28.6)])
        radius = 1
        inclination = math.radians(51)
        azimuth = 0
        psi = 0
        beta = 0
        self.state =tf.Variable([azimuth, radius, inclination, psi, beta])
        self.acc = tf.constant(0.00351598)
        self.gain = tf.constant(0.0024262)
        self.time_step = tf.constant(0.1)
        self.Time = tf.Variable(0, dtype=float)
    def reset(self):
        radius = 1
        inclination = math.radians(51)
        azimuth = 0
        psi = 0
        beta = 0
        self.state = tf.Variable([azimuth, radius, inclination, psi, beta])
        state = self.state
        return state, False

    @tf.function
    def step(self, action):
        h = self.time_step
        done = False
        az, radius, inclination, psi, beta = tf.unstack(self.state)

        err_1 = tf.norm(self.state[1:3]-self.end_point)

        dpsi = beta
        dbeta = tf.cast(action-1, float) * self.gain
        daz = (1 / (radius ** 1.5))
        dr = 2 * self.acc * (radius ** 1.5) * tf.cos(psi)
        di = self.acc * tf.sin(psi) * (radius ** 0.5) * tf.cos(az)

        self.dstate = tf.stack([daz, dr, di, dpsi, dbeta])

        self.state.assign_add(self.dstate*h)
        err_2 = tf.norm(self.state[1:3] - self.end_point)

        reward = err_1 - err_2
        self.Time.assign_add(h)

        if self.Time > 250:
            done = True
        if abs(self.state[3]) > 2.3562:
            done = True
        if self.state[1] < 1:
            done = True


        return self.state, reward, done, self.Time