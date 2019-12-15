import sys
sys.path.append('./model')
sys.path.append('./game')
import torch
import cv2
import random
import os
import game
import numpy as np
from dqn import DQN, resized_img, actions, weight_init
from game import best_score
from collections import deque
from tensorboardX import SummaryWriter

# random.seed(1e6)

# [80, 80, 3] -> [80, 80]
def bgr2gray(rgb_img):
    gray_img = cv2.cvtColor(cv2.resize(rgb_img, resized_img), cv2.COLOR_BGR2GRAY)
    _, gray_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    gray_img = np.array(gray_img, dtype = np.float32)
    return gray_img

def data2tensor(data, device):
    return torch.tensor(data, dtype = torch.float32, device = device)

def start(mode = 'train'):
    if mode not in ['train', 'test']:
        raise ValueError('Unknown mode!')

    '''
        Configuration
    '''
    observe_steps = 1000
    memory_size = 50000
    epoch = 2000000
    use_pretrained_model = False
    save_model_path = './ckpt/model5/'
    save_model_name = 'model.pkl'
    pretrained_model_path = save_model_path + save_model_name
    log_path = './log/log5/1.0/'
    init_epsilon = 0.1
    final_epsilon = 0.0001
    frame_per_action = 1
    epsilon = init_epsilon if mode is 'train' else 0
    init_learning_rate = 1e-2
    batch_size = 32
    start_epoch = 1
    gamma = 0.99
    history_size = 4

    if os.path.exists(save_model_path) is False:
        os.makedirs(save_model_path)
    if os.path.exists(log_path) is False:
        os.makedirs(log_path)

    '''
        Build the network
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = DQN().to(device).float()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = init_learning_rate, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 500000, gamma = 0.1)
    # Read the pretrained model
    if use_pretrained_model:
        checkpoint = torch.load(pretrained_model_path)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        # Cuda
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        print("Load the pretrained model from %s successfully!" % pretrained_model_path)
    else:
        weight_init(net)
        print("First time training!")
    
    '''
        Data structures
    '''
    writer = SummaryWriter(log_path)
    flappybird = game.GameState()
    memory_replay = deque()
    # No action
    aidx = 0
    action = np.zeros([2], dtype = np.float32)
    action[aidx] = 1
    img, reward, terminate = flappybird.frame_step(aidx)
    img = bgr2gray(img)
    # img_seq: 4x80x80
    img_seq = np.stack([img for _ in range(history_size)], axis = 0)

    '''
        Start the game
    '''
    # Train DQN
    if mode is 'train':
        stage = 'OBSERVE'
        print('Start training...')
        for e in range(start_epoch, epoch + start_epoch):
            img_seq_ts = data2tensor(img_seq, device).unsqueeze(0)
            pred = net(img_seq_ts)

            # Take an action
            idx, action = 0, np.zeros([actions], dtype = np.float32)
            if e % frame_per_action == 0:
                # Epsilon greedy policy
                if random.random() <= epsilon:
                    idx = random.randint(0,1)
                else:
                    idx = torch.argmax(pred, dim = 1).item()
            else:
                idx = 0
            action[random.randint(0,1)] = 1

            # Scale down epsilon
            epsilon -= (init_epsilon - final_epsilon) / epoch

            # Run an action
            img_next, reward, terminate = flappybird.frame_step(idx)
            # img_next = data2tensor(bgr2gray(img_next), device).unsqueeze(0)
            img_next = bgr2gray(img_next)
            img_seq_next = np.stack([img_next, img_seq[0], img_seq[1], img_seq[2]], axis = 0)

            # Update the memory
            memory_replay.append([img_seq, img_seq_next, action, reward, terminate])
            if len(memory_replay) > memory_size:
                memory_replay.popleft()

            if e <= start_epoch + observe_steps and e % 1000 == 0:
                print('Finish %d observations!' % e)

            # Train after observation
            if e > start_epoch + observe_steps:
                stage = 'TRAINING'
                batch = random.sample(memory_replay, batch_size)
                img_seq_b, img_seq_next_b, action_b, reward_b, terminate_b = zip(*batch)

                # img_seq_b_ts = data2tensor(np.stack([ib for ib in img_seq_b], axis = 0), device)
                img_seq_b_ts = torch.from_numpy(np.stack([ib for ib in img_seq_b], axis = 0)).to(device)
                # img_seq_next_b_ts = data2tensor(np.stack([ib for ib in img_seq_next_b], axis = 0), device)
                img_seq_next_b_ts = torch.from_numpy(np.stack([ib for ib in img_seq_next_b], axis = 0)).to(device)
                # action_b_ts = data2tensor(action_b, device)
                action_b_ts = torch.from_numpy(np.array(list(action_b))).to(device)
                out = net(img_seq_b_ts)
                out_next = net(img_seq_next_b_ts)
                # y_b = torch.tensor([reward_b[bi] if terminate_b[bi] else reward_b[bi] + gamma * torch.max(out_next).item() for bi in range(batch_size)]).to(device)
                y_b = []
                for r, t, p in zip(reward_b, terminate_b, out_next):
                    if t:
                        y_b.append(r)
                    else:
                        y_b.append(r + gamma * torch.max(p).item())
                # y_b = data2tensor(y_b, device)
                y_b = torch.from_numpy(np.array(y_b, dtype = np.float32)).to(device)
                q_value_b = torch.sum(out * action_b_ts, dim = 1)

                # Calculate loss and back propagation
                optimizer.zero_grad()
                loss = loss_func(q_value_b, y_b)
                loss.backward()
                optimizer.step()

                # Print information
                print('Epoch %d: stage = %s, loss = %.6f, Q_max = %.6f, action = %d, reward = %.3f' % (e, stage, loss.item(), torch.max(pred).item(), idx, reward))
                writer.add_scalar('Train/Loss', loss.item(), e)
                writer.add_scalar('Train/Epsilon', epsilon, e)
                writer.add_scalar('Train/Reward', reward, e)
                writer.add_scalar('Train/Q-Max', torch.max(pred).item(), e)

            img_seq = img_seq_next

            # Save model
            if e % 10000 == 0:
                states = {
                    'epoch' : e,
                    'state_dict' : net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }
                torch.save(states, save_model_path + save_model_name)
                print('Save the model at epoch %d successfully!' % e)

            scheduler.step()            

    # Test DQN
    else:
        pass
    


if __name__ == '__main__':
    start(mode = 'train')
