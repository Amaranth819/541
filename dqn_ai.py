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
from torch.autograd import Variable

# random.seed(1e6)

def bgr2gray(rgb_img):
    gray_img = cv2.cvtColor(cv2.resize(rgb_img, resized_img), cv2.COLOR_RGB2GRAY)
    _, gray_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    gray_img = np.array(gray_img, dtype = np.float32) / 255
    return gray_img

def data2tensor(data, device):
    return torch.tensor(data, dtype = torch.float32, device = device)

# for expectimax
def getExpectimax(ex_max, height_proba, s_mat, h_mat, a_mat, step):
    # ex_max: expectiMax
    # # the structure of each element in ex_max: 
    # # [height, # of occurrance, # up survival, # of up not survival, # down survival, # of down not survival]
    # i is the row number
    fix = 100
    for i in range(len(s_mat)):
        row_len = len(s_mat[i])
        # j is the col number
        row_lens = random.sample(range(row_len), fix) if row_len > fix else range(row_len)
        for j in row_lens:
            for k in range(len(ex_max)):
                if ex_max[k][0] == h_mat[i][j]:
                    ex_max[k][1] += 1
                    # check whether the bird is alive after # of step
                    if j + step < len(s_mat[i]):
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
                    ex_max = np.append(ex_max,[[h_mat[i][j],10,5,0,5,0]],axis = 0)
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
    # ex_max = np.delete(ex_max,0,0)
    # print(ex_max)
    # calculate the survival prob at specific height for doing different action

    # for row in ex_max:
    #     if row[1] != 0:
    #         height_proba = np.append(height_proba,[[row[0],row[2]/row[1],row[4]/row[1]]],axis = 0)
    #     else:
    #         height_proba = np.append(height_proba,[[row[0],0,0]],axis = 0)
    
    # height_proba = np.delete(height_proba,0,0)
    # print(height_proba)
    return ex_max, height_proba
  
           
def getHeightDecision(ex_max, height):
    # print(height_proba)
    result = np.array([1, 0], dtype = np.float32)
    for row in ex_max:
        up_survival = row[2] / row[1]
        up_dead = row[3] / row[1]
        down_survival = row[4] / row[1]
        down_dead = row[5] / row[1]
        if row[0] == height:
            if up_survival >= down_survival and up_dead <= down_dead:
                result = np.array([0,1], dtype = np.float32)
            else:
                result = np.array([1,0], dtype = np.float32)
            break
    # By default
    return result


def start(mode = 'train'):
    if mode not in ['train', 'test']:
        raise ValueError('Unknown mode!')

    '''
        Configuration
    '''
    observe_steps = 5
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
    init_learning_rate = 1e-6
    batch_size = 2
    start_epoch = 1
    gamma = 0.99
    history_size = 4
    max_window_size = 128
    random.seed(1e6)
    use_expectimax = True

    # for expectiMax 
    # ex_max: detail data of expectiMax
    ex_max = np.zeros((1,6))
    # height_proba: the probability of up action and down action.
    height_proba = np.zeros((1,3))
    # 
    step = 60

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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [500, 2000], gamma = 0.5)
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
    # img_seq: 4x84x84
    img_seq = np.stack([img for _ in range(history_size)], axis = 0)

    '''
        Start the game
    '''
    # Train DQN
    if mode is 'train':
        net.train()
        stage = 'OBSERVE'
        print('Start training...')
        for e in range(start_epoch, epoch + start_epoch):
            per_game_memory = deque()
            while True:
                # img_seq_ts = data2tensor(img_seq, device).unsqueeze(0)
                img_seq_ts = Variable(torch.from_numpy(img_seq).unsqueeze(0).to(device))
                pred = net(img_seq_ts)

                # Take an action
                idx, action = 0, np.zeros([actions], dtype = np.float32)
                if e % frame_per_action == 0:
                    if stage is 'OBSERVE':
                        idx = 0 if random.random() < 0.9 else 1
                    else:
                        # Epsilon greedy policy
                        if random.random() <= epsilon:
                            idx = 0 if random.random() < 0.9 else 1
                        else:
                            idx = torch.argmax(pred, dim = 1).item()
                else:
                    idx = 0
                action[idx] = 1

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
                    # Get all history of 'batch_size' games
                    batch = random.sample(memory_replay, batch_size)

                    # Get the game history
                    img_seq_b, img_seq_next_b, action_b, reward_b, terminate_b, curr_height_b = [], [], [], [], [], []
                    for b in batch:
                        ib, inb, ab, rb, tb, chb = zip(*b)
                        img_seq_b.append(ib)
                        img_seq_next_b.append(inb)
                        action_b.append(ab)
                        reward_b.append(rb)
                        terminate_b.append(tb)
                        curr_height_b.append(chb)
                    reward_b_window = np.concatenate(reward_b, axis = 0)
                    terminate_b_window = np.concatenate(terminate_b, axis = 0)

                    # Randomly sample if there are too many frames in total.
                    all_frames_num = terminate_b_window.shape[0]
                    window = np.array(random.sample(list(range(all_frames_num)), max_window_size)) if all_frames_num > max_window_size else np.array(list(range(all_frames_num)))
                    reward_b_window = np.take(reward_b_window, window, axis = 0)
                    terminate_b_window = np.take(terminate_b_window, window, axis = 0)

                    # Get the image states
                    img_seq_b_ts = Variable(torch.from_numpy(np.take(np.concatenate(img_seq_b, axis = 0), window, axis = 0)).to(device))
                    img_seq_next_b_ts = Variable(torch.from_numpy(np.take(np.concatenate(img_seq_next_b, axis = 0), window, axis = 0)).to(device))

                    if use_expectimax is True:
                        s_mat = terminate_b[:][:-1]
                        h_mat = curr_height_b[:][:-1]
                        a_mat = action_b[:][:-1]
                        ex_max, height_proba = getExpectimax(ex_max, height_proba, s_mat, h_mat, a_mat, step)
                        action_b_ts = torch.from_numpy(getHeightDecision(ex_max, curr_height)).to(device)
                    else:
                        action_b_window = np.take(np.concatenate(action_b, axis = 0), window, axis = 0)
                        action_b_ts = torch.from_numpy(np.array(action_b_window)).to(device)

                    # Predict
                    out = net(img_seq_b_ts)
                    out_next = net(img_seq_next_b_ts)
                    # Calculate y and q value
                    y_b = []
                    for r, t, p in zip(reward_b_window, terminate_b_window, out_next):
                        if t:
                            y_b.append(r)
                        else:
                            y_b.append(r + gamma * torch.max(p).item())
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

                scheduler.step()   

                if terminate:
                    break    

            # Save model
            if e % 50 == 0:
                states = {
                    'epoch' : e,
                    'state_dict' : net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }
                torch.save(states, save_model_path + save_model_name)
                print('Save the model at epoch %d successfully!' % e)

            # Update the memory
            memory_replay.append(per_game_memory)
            if len(memory_replay) > memory_size:
                memory_replay.popleft()     

    # Test DQN
    else:
        pass
    


if __name__ == '__main__':
    start(mode = 'train')
