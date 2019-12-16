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

# for expectimax
def getExpectimax(ex_max,height_proba,s_mat, h_mat, a_mat, step):
    # ex_max: expectiMax
    # # the structure of each element in ex_max: 
    # # [height, # of occurrance, # up survival, # of up not survival, # down survival, # of down not survival]
    # i is the row number
    for i in range(len(s_mat)):
    row_len = len(s_mat[i])
    # j is the col number
    for j in range(row_len):
        for k in range(len(ex_max)):
        if ex_max[k][0] == h_mat[i][j]:
            ex_max[k][1] += 1
            # check whether the bird is alive after # of step
            if j + step <= len(s_mat[i]):
            # up action
            if a_mat[i][j][0] == 0:
                ex_max[k][2] += 1
            # down action
            else:
                ex_max[k][4] += 1
            else:
            if a_mat[i][j][0] == 0:
                ex_max[k][3] += 1
            else:
                ex_max[k][5] += 1
            break
        elif k == len(ex_max)-1:
            # print("Current height: ")
            # print(h_mat[i][j])
            ex_max = np.append(ex_max,[[h_mat[i][j],0,0,0,0,0]],axis = 0)
            ex_max[-1][1] += 1
            if j + 10 <= len(s_mat[i]):
            if a_mat[i][j][0] == 0:
                ex_max[-1][2] += 1
            else:
                ex_max[-1][4] += 1
            else:
            if a_mat[i][j][0] == 0:
                ex_max[-1][3] += 1
            else:
                ex_max[-1][5] += 1
    ex_max = np.delete(ex_max,0,0)
    # print(ex_max)
    # calculate the survival prob at specific height for doing different action
    for row in ex_max:
    if row[1] != 0:
        height_proba = np.append(height_proba,[[row[0],row[2]/row[1],row[4]/row[1]]],axis = 0)
    else:
        height_proba = np.append(height_proba,[[row[0],0,0]],axis = 0)
    height_proba = np.delete(height_proba,0,0)
    # print(height_proba)
    return ex_max,height_proba
  
           
def getHeightDecision(height_proba,height):
    # print(height_proba)
    for row in height_proba:
        if row[0] == height:
            if row[1] > row[2]:
                result = np.array([0,1], dtype = np.float32)
            else:
                result = np.array([1,0], dtype = np.float32)
        return result

    # By default
    return np.array([1, 0], dtype = np.float32)


def start(mode = 'train'):
    if mode not in ['train', 'test']:
        raise ValueError('Unknown mode!')

    '''
        Configuration
    '''
    observe_steps = 20
    memory_size = 1000
    epoch = 5000 # Game 
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
    batch_size = 16
    start_epoch = 1
    gamma = 0.99
    history_size = 4

    # for expectiMax 
    # ex_max: detail data of expectiMax
    ex_max = np.zeros((1,6))
    # height_proba: the probability of up action and down action.
    height_proba = np.zeros((1,3))
    # 
    step = 10

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10000, gamma = 0.1)
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
    curr_height = flappybird.playery
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
            per_game_memory = deque()
            while True:
                # img_seq_ts = data2tensor(img_seq, device).unsqueeze(0)
                img_seq_ts = torch.from_numpy(img_seq).unsqueeze(0).to(device)
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
                curr_height = flappybird.playery
                # img_next = data2tensor(bgr2gray(img_next), device).unsqueeze(0)
                img_next = bgr2gray(img_next)
                img_seq_next = np.stack([img_next, img_seq[0], img_seq[1], img_seq[2]], axis = 0)

                # Update the memory
                # memory_replay.append([img_seq, img_seq_next, action, reward, terminate])
                per_game_memory.append([img_seq, img_seq_next, action, reward, terminate, curr_height])
                # if len(memory_replay) > memory_size:
                #     memory_replay.popleft()

                if e <= start_epoch + observe_steps and e % 1000 == 0:
                    print('Finish %d observations!' % e)

                # Train after observation
                if e > start_epoch + observe_steps:
                    stage = 'TRAINING'
                    batch = random.sample(memory_replay, batch_size)
                    # img_seq_b, img_seq_next_b, action_b, reward_b, terminate_b, curr_height_b = zip(*batch)
                    img_seq_b = [b[0] for b in batch]
                    img_seq_next_b = [b[1] for b in batch]
                    action_b = [b[2] for b in batch]
                    reward_b = [b[3] for b in batch]
                    terminate_b = [b[4] for b in batch]
                    curr_height_b = [b[5] for b in batch]

                    for i in range(len(action_b)):
                        s_mat = terminate_b[i][:-1]
                        h_mat = curr_height_b[i][:-1]
                        a_mat = action_b[i][:-1]
                        ex_max, height_proba = getExpectimax(ex_max, height_proba, s_mat, 
                                                                h_mat, a_mat, step)


                    # img_seq_b_ts = data2tensor(np.stack([ib for ib in img_seq_b], axis = 0), device)
                    img_seq_b_ts = torch.from_numpy(np.stack([ib for ib in img_seq_b], axis = 0)).to(device)
                    # img_seq_next_b_ts = data2tensor(np.stack([ib for ib in img_seq_next_b], axis = 0), device)
                    img_seq_next_b_ts = torch.from_numpy(np.stack([ib for ib in img_seq_next_b], axis = 0)).to(device)
                    # action_b_ts = data2tensor(action_b, device)
                    # action_b_ts = torch.from_numpy(np.array(list(action_b))).to(device)

                    # get the action according to the current height and expectimax
                    action_b_ts = torch.from_numpy(getHeightDecision(height_proba, curr_height)).to(device)
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

                if terminate:
                    break    

            memory_replay.append(per_game_memory)
            if len(memory_replay) > memory_size:
                memory_replay.popleft()     

    # Test DQN
    else:
        pass
    


if __name__ == '__main__':
    start(mode = 'train')
