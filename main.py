import numpy as np
from Agent import Agent
from Enviroment import Enviroment
env = Enviroment()
agent = Agent()

max_Score = 0
score_list = list()


for episode in range(2000):
    print("Episode: ", episode+1)
    print("Max Score", max_Score)
    state, done = env.reset()
    Score = 0

    while not done:
        action = agent.select_action(state)
        state_, reward, done, time = env.step(action)
        agent.remember(state, action, state_, reward, done)
        state = state_
        Score += reward
        agent.learn()
        agent.target_train()
    score_list.append(Score)    
    if Score >= max_Score:
        max_Score = Score
    print(state_)
    print("Score: ", Score)
    print("Time:", time)

    out = np.array(score_list).transpose()
    np.savetxt('res.txt', out)