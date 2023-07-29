import numpy as np
import matplotlib.pyplot as plt
import math
from DQN import Agent
import tensorflow as tf


class Calcs():
    def __init__(self):
        self.inc_list = list()
        self.r_list = list()
        self.l_list = list()
        self.p_list = list()
        self.time_list = list()
        self.inc0 = 51
        self.inc_k = 28.6
        self.r_k = 46
        self.agent = Agent()
    def get_radius(self):
        return self.r_list

    def get_angles(self):
        return self.p_list
    def get_inc(self):
        return self.inc_list
    def get_control(self):
        return self.l_list
    def get_time(self):
        return self.time_list

    def fly(self):
        r = 1
        phi = 0
        beta = 0
        lam = 0

        v = 0
        omega = 0
        inc = math.radians(self.inc0)
        inc_k = math.radians(self.inc_k)
        r_k = self.r_k
        self.r_list.clear()
        self.inc_list.clear()
        self.p_list.clear()
        self.l_list.clear()
        self.time_list.clear()
        T=0
        acc = 0.00351598
        S = 0

        dt = 0.1
        while r < r_k:
            if T > 300:
                break
            phi_n = phi % (2 * np.pi)
            self.r_list.append(r)#*8371)
            self.time_list.append(v)#*6900.5)
            self.p_list.append(math.degrees(phi))
            self.l_list.append(math.degrees(lam))
            self.inc_list.append(math.degrees(inc))

            prev_err = np.sqrt((inc - inc_k) ** 2 + (r - r_k) ** 2)

            inp = np.array([[r, inc, phi_n, lam, beta]])

            u = np.argmax(self.agent.select_action(inp).numpy())-1


            dlam = beta
            dbeta = u*0.001
            dp = (1 / (r ** 1.5))
            dr = 2*acc*(r**1.5)*math.cos(lam)
            di = acc*math.sin(lam)*(r**0.5)*np.cos(phi)
            dom = acc*math.sin(lam)*(r**0.5)*np.sin(phi)
            dV = acc

            lam += dlam*dt
            beta += dbeta*dt
            phi += dp*dt
            r += dr * dt
            inc += di * dt
            omega += dom * dt
            v += dV*dt
            T += dt

            err = np.sqrt((inc - inc_k) ** 2 + (r - r_k) ** 2)

            S += (prev_err-err)
        err =np.sqrt((inc-inc_k)**2+(r-r_k)**2)
        #S -= err
        print("err: ", err)
        print("omega: ", math.degrees(omega))
        print("SumQ: ",S)
        return err
    def train(self,episode):
        r = 1
        phi = 0
        beta = 0
        lam = 0
        u = 0
        v = 0
        omega = 0
        inc = math.radians(self.inc0)
        inc_k = math.radians(self.inc_k)
        r_k = self.r_k
        T = 0
        acc = 0.0035159

        dt = 0.1
        N = 0
        while r < r_k:
            N+=1
            if T > 300:
                break
            phi = phi % (2 * np.pi)
            prev_err = np.sqrt((inc - inc_k) ** 2 + (r - r_k) ** 2)
            inp = np.array([[r, inc, omega, lam, beta]])
            state = inp[0].tolist()



            action = np.argmax(self.agent.select_action(inp).numpy())
            eps = np.random.randint(0, 100, 1)
            if eps <= 10:
                action = np.random.randint(0, 2, 1)[0]

            u = action - 1
            dlam = beta
            dbeta = u * 0.001
            dp = (1 / (r ** 1.5))
            dr = 2 * acc * (r ** 1.5) * math.cos(lam)
            di = acc * math.sin(lam) * (r ** 0.5) * np.cos(phi)
            dom = acc * math.sin(lam) * (r ** 0.5) * np.sin(phi)
            dV = acc

            lam += dlam * dt
            beta += dbeta * dt
            phi += dp * dt
            r += dr * dt
            inc += di * dt
            omega += dom * dt
            v += dV * dt
            T += dt

            err = np.sqrt((inc - inc_k) ** 2 + (r - r_k) ** 2)
            reward = prev_err-err

            next_state = [r, inc, omega, lam, beta]

            if np.random.randint(0,100,1)[0] < 20:
                self.agent.add_to_replay([state,action,reward,next_state, err])

        self.agent.train()


calcs = Calcs()

err = 45

for episode in range(2000):
    calcs.train(episode)
    err = calcs.fly()

angle = calcs.get_angles()
time = calcs.get_time()
radius = calcs.get_radius()
inclination = calcs.get_inc()

plt.plot(time, radius)
plt.xlabel("Характеристическая скорость")
plt.ylabel("Восота орбиты, км")
plt.show()

plt.plot(time, inclination)
plt.xlabel("Характеристическая скорость")
plt.ylabel("Наклонение орбиты, град")
plt.show()

plt.plot(time, calcs.get_control())
plt.xlabel("Характеристическая скорость")
plt.ylabel("Управляющий сигнал, град")
plt.show()


