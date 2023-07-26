import numpy as np
import matplotlib.pyplot as plt
import math
from DQN import MyModel


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
        self.mumodel = MyModel()
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


    def equs(self):
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
        self.r_list.clear()
        self.inc_list.clear()
        self.p_list.clear()
        self.l_list.clear()
        self.time_list.clear()
        T=0
        acc = 0.00351598
        while r < r_k:



            self.r_list.append(r)#*8371)
            self.time_list.append(v)#*6900.5)
            self.p_list.append(math.degrees(phi))
            self.l_list.append(math.degrees(lam))
            self.inc_list.append(math.degrees(inc))

            dt = np.pi/1000

            dp = (1 / (r ** 1.5))
            dr = 2*acc*(r**1.5)*math.cos(lam)
            di = acc*math.sin(lam)*(r**0.5)*np.cos(phi)
            dom = acc*math.sin(lam)*(r**0.5)*np.sin(phi)
            dV = acc

            phi += dp*dt
            r += dr * dt
            inc += di * dt
            omega += dom * dt
            v += dV*dt
            T += dt
        err =np.sqrt((inc-inc_k)**2+(r-r_k)**2)
        print("err: ", err)
        print("omega: ", math.degrees(omega))
        print("Time: ",T)



calcs = Calcs()



calcs.equs()

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
plt.show()
