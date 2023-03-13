import gym
import numpy as np
from DDQN import Agent as DDQN
from DQN import Agent as DQN
from utils import plot_learning
import hickle

# env = gym.make("LunarLander-v2", render_mode='rgb_array')
# env = gym.wrappers.RecordVideo(env, 'video_dd', episode_trigger=lambda x: x % 5 == 0)
env = gym.make("LunarLander-v2")

agent = DQN(input_dim=(8,), n_actions=4, lr=0.001, gamma=0.99, epsilon_end=0.01)
# agent = DQN(input_dim=(8,), n_actions=4, lr=0.001, gamma=0.99, epsilon_end=0.01)
# agent.load_model('model/dqn_model_100.h5')

max_episodes = 1000
max_steps = 1000
scores = []
avg_scores = []

for episode in range(max_episodes):

    observation, info = env.reset()
    step = 0
    score = 0

    while step < max_steps:
        action = agent.make_action(observation)

        observation_, reward, terminated, truncated, info = env.step(action)

        agent.store_data(observation, action, reward, observation_, terminated)
        agent.train(64)

        observation = observation_
        score += reward

        if terminated or truncated:
            break

        step += 1

    scores.append(score)
    avg_scores.append(np.mean(scores[max(0, episode - 100):(episode + 1)]))

    print('Episode:{} Score:{} AVG Score:{}'.format(
        episode+1, scores[episode], avg_scores[episode]))

    if (episode + 1) % 100 == 0:
        print('.... model saved ....')
        agent.save_model(f'model/ddqn_model_{episode + 1}.h5')
    if avg_scores[-1] > 200:
        break

agent.save_model('model/dqn_model_final.h5')
plot_learning(scores, avg_scores, 'Wyniki Agenta', 'plot/car_racing_dqn.png')
hickle.dump([scores, avg_scores], 'data/scores_dqn.hkl')
env.close()
