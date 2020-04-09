import torch as T
import argparse
from model import ActorCritic
from env import ConnectX


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, help="Weights to load", required=True)
    parser.add_argument("-n", "--num", type=int, help="Number of games", default=1)
    parser.add_argument("--rnd", type=bool, help="Play against random agent (else against negamax)", default=False)
    opts = parser.parse_args()


    # Autodetect CUDA
    use_cuda = T.cuda.is_available()
    device   = T.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    HIDDEN_SIZE = 256
    env = ConnectX(switch_prob=0.5, random_agent=opts.rnd, test_mode=True)
    model=ActorCritic(env.observation_space.n, env.action_space.n, HIDDEN_SIZE)
    model.load_state_dict(T.load(opts.weights))


    total_reward = 0
    for _ in range(opts.num): 
        state = env.reset()
        done = False
        
        while not done:
            state = T.FloatTensor(state.board).unsqueeze(0).to(device)
            dist, _ = model(state)
            dist_space = dist.sample()

            action = T.argmax(dist_space, dim=1, keepdim=True).cpu().numpy()[0]

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += 1 if reward==1 else 0
            env.render(mode='ansi')

    win_ratio = total_reward/float(opts.num)*100.0
    print(f'Played {opts.num} games and won {win_ratio} %')