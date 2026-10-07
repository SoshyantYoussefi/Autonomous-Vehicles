# Information
Trained for 8000 episodes with repeated exploration phases. Continuation
of the 2021_07_30-06_19_03 model.
```
class Q_NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(25, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.ReLU()
        )

    def forward(self, s):
        q = self.layers(self.flatten(s))
        return q

def explore_exploit(k, eps):
    if type(eps) == tuple:
        p0, p1, k1 = eps
        p = (p0 - p1) * math.exp(-k / k1) + p1
    else:
        p = eps
    return p

env = load_environment("../rl_agents/scripts/configs/HighwayEnv/env.json")
agent = DQN_Agent(env, Q_NN, explore_exploit=lambda k: explore_exploit(k, (1.0, 0.05, 200)),
                  gamma=0.8, batch_size=32)
agent.train(1500, display=False)
agent.explore_exploit = lambda k: explore_exploit(k, (1.0, 0.05, 200))
agent.train(1500, display=False)
agent.explore_exploit = lambda k: explore_exploit(k, (1., 0.01, 200))
agent.train(1500, display=False)
agent.explore_exploit = lambda k: explore_exploit(k, (1., 0.01, 200))
agent.train(1750, display=False)
agent.explore_exploit = lambda k: explore_exploit(k, (1., 0.01, 200))
agent.train(1750, display=False)

agent.save()


```
