from Enviroment import Enviroment
from  Agent import Agent
env = Enviroment()
agent = Agent()




for episode in range(2000):
    print("Episode: ", episode+1)
    state, done = env.reset()
    Score = 0

    while not done:

        action = agent.select_action(state)
        state_, reward, done, time = env.step(action)
        agent.remember(state, action, state_, reward, done)
        state = state_
        Score += reward
        agent.learn()
    print(state_)
    print("Score: ", Score)
    print("Time:", time)