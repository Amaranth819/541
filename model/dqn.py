import torch
import torch.nn as nn

history_size = 4 
resized_img = [80, 80]
actions = 2 # action[0] = 1 means no flapping and action[1] = 1 means flapping.

class DQN(nn.Module):
    '''
        Input: history_size * 1 (Grayscale) * 80 * 80
        Output: 1 * 2
    '''
    def __init__(self):
        super(DQN, self).__init__()

        '''
            Build the network
        '''
        # Encoder
        # hs*1*80*80 -> hs*32*20*20
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 4, stride = 4, padding = 0)
        # hs*32*20*20 -> hs*64*10*10
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1)
        # hs*64*10*10 -> hs*64*5*5
        self.down_sample1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace = True)
        )
        # hs*64*5*5 -> hs*64*2*2
        self.down_sample2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1, inplace = True)
        )

        # GRU
        # hs*64*2*2 -> hs*1*256
        # hs*1*256 -> hs*1*256 (gru output), 1*1*256 (hidden state)
        self.gru_hidden_size = 256
        self.gru_layer_num = 1
        self.gru = nn.GRU(256, self.gru_hidden_size, self.gru_layer_num, dropout = 0)

        # Attentional-based net

        # FC
        self.fc_relu = nn.Sequential(
            nn.Linear(self.gru_hidden_size, actions),
            nn.LeakyReLU(0.1, inplace = True)
        )
        self.softmax = nn.Softmax()
    
    def init_hidden(self):
        if torch.cuda.is_available():
            return torch.zeros(self.gru_layer_num, 1, self.gru_hidden_size).cuda()
        else:
            return torch.zeros(self.gru_layer_num, 1, self.gru_hidden_size)

    def forward(self, x):
        # Encoder
        x = self.conv2(self.conv1(x))
        x = self.down_sample1(x)
        x = self.down_sample2(x)

        # GRU
        hidden = self.init_hidden()
        x, hidden = self.gru(x.view(-1, 1, 256), hidden)

        # FC
        x = self.fc_relu(torch.squeeze(x[-1]))
        x = self.softmax(x)

        return x

if __name__ == '__main__':
    net = DQN()
    x = torch.ones((history_size, 1, 80, 80))
    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()
    y = net(x)
    print(x.shape)
    print(y.shape)