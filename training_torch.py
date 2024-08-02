# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# %%
file_path = 'data/data1_compressed/data0.json'
df = pd.read_json(file_path)

# %%
df.rename(columns={0: 'ram_observation', 1: 'reward'}, inplace=True)

df['ram_observation'] = df['ram_observation'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
df['reward'] = df['reward'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

df.head()


# %%
import torch
import torch.nn as nn
import torch.optim as optim

# %%
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(129, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)            
        return x

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.device(device)
print(f"Using {device} device")
torch.manual_seed(123455)

# %%
model1 = DeepQNetwork()
model2 = DeepQNetwork()


