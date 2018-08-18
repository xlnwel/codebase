import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from copy import deepcopy


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        basic_channels = 64
        n_residual = 3
        
        self.input_layer = nn.Conv2d(3, basic_channels, 3, padding=1)
        
        # residual modules:
        self.res_net = nn.ModuleList()
        # layers decreasing spatial dimension and increasing channels 
        self.deepen_layer = nn.ModuleList()
        
        for i in range(n_residual):
            n_channels = 2**i * basic_channels
            bn1 = nn.BatchNorm2d(n_channels)
            conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
            bn_conv1 = nn.ModuleList([bn1, conv1])
            bn2 = nn.BatchNorm2d(n_channels)
            conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
            bn_conv2 = nn.ModuleList([bn2, conv2])
            res = nn.ModuleList([bn_conv1, bn_conv2])
            self.res_net.append(res)
        
            bn = nn.BatchNorm2d(n_channels)
            conv = nn.Conv2d(n_channels, 2 * n_channels, 3, stride=2, padding=1)
            bn_conv = nn.ModuleList([bn, conv])
            self.deepen_layer.append(bn_conv)
        
        self.fc = nn.Linear(2**n_residual * basic_channels, 4)

        self._reset_params()
    
    def _reset_params(self):
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='relu')
        nn.init.constant_(self.input_layer.bias, 0)

        for bn_conv1, bn_conv2 in self.res_net:
            conv1, conv2 = bn_conv1[1], bn_conv2[1]
            nn.init.kaiming_normal_(conv1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(conv2.weight, nonlinearity='relu')
            nn.init.constant_(conv1.bias, 0)
            nn.init.constant_(conv2.bias, 0)

        for _, conv in self.deepen_layer:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
            nn.init.constant_(conv.bias, 0)
        
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        for i, res in enumerate(self.res_net):
            y = x
            for bn_conv in res:
                bn, conv = bn_conv
                y = bn(y)
                y = F.relu(y)
                y = conv(y)
            x = x + y
            
            bn, conv = self.deepen_layer[i]
            x = bn(x)
            x = F.relu(y)
            x = conv(x)
        # flatten
        x = x.mean(2).mean(2)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        
        return x


class ReplayBuffer():
    def __init__(self, sample_size, buffer_size=int(1e4)):
        self.buffer = deque(maxlen=buffer_size)
        self.sample_size = sample_size
        self.experience = namedtuple('experience', field_names=('state', 'action', 'reward', 'next_state', 'done'))
        
    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
        
    def sample(self):
        experiences = random.sample(self.buffer, self.sample_size)
        
        # divide into batches
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, alpha=1e-1, gamma=0.99, epsilon=0.1, tau=1e-3, action_size=action_size, batch_size=2):
        self.Q = DQN().to(device)
        self.Q_target = deepcopy(self.Q).to(device)
        self.buffer = ReplayBuffer(sample_size=batch_size)
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.alpha)
    
    def _transpose(self, state):
        # convert HWC to CHW
        state = np.transpose(state, (0, 3, 1, 2))
        
        return state
    
    def act(self, state):
        state = self._transpose(state)
        state = torch.from_numpy(state).float().to(device)
        self.Q.eval()
        with torch.no_grad():
            action_values = self.Q(state)
        self.Q.train()
        
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(action_values.cpu().numpy())
        
    def step(self, state, action, reward, next_state, done):
        state = self._transpose(state)
        next_state = self._transpose(next_state)
        self.buffer.add(state, action, reward, next_state, done)
        
        if len(self.buffer) > 100 + self.buffer.sample_size:
            experiences = self.buffer.sample()
            self._update_net(experiences)
            
    def _update_net(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        targets = rewards + (1 - dones) * self.gamma * self.Q_target(next_states).detach().max(1)[0].reshape((-1, 1))
        predictions = self.Q(states).gather(1, actions)
        
        loss = F.mse_loss(predictions, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._moving_average()
    
    def _moving_average(self):
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)