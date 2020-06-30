import numpy as np


'''
Blockudoku 환경에 쓰일 블록을 나타내는 object

b_dict      : 블록의 종류를 나열해 둔 dictionary
lens        : 블록을 sample할 때 쓰이는 attribute
block_array : 블록의 모양을 나타내는 numpy array
is_used     : 블록이 사용되었음을 환경에 알려주는 attribute
'''
class Block:
    def __init__(self):
        self.b_dict = dict()
        self.b_dict[1] = [[(2, 2)]]
        self.b_dict[2] = [[(1, 2), (2, 2)],
                          [(1, 3), (2, 2)],
                          [(2, 1), (2, 2)],
                          [(2, 2), (3, 3)]]
        self.b_dict[3] = [[(1, 2), (2, 2), (3, 2)],
                          [(2, 2), (2, 3), (3, 2)],
                          [(1, 2), (2, 2), (2, 3)],
                          [(1, 2), (2, 1), (2, 2)],
                          [(2, 1), (2, 2), (3, 2)],
                          [(1, 3), (2, 2), (3, 1)],
                          [(2, 1), (2, 2), (2, 3)],
                          [(1, 1), (2, 2), (3, 3)]]
        self.b_dict[4] = [[(0, 2), (1, 2), (2, 2), (3, 2)],
                          [(2, 2), (2, 3), (3, 2), (4, 2)],
                          [(1, 2), (2, 2), (2, 3), (3, 2)],
                          [(0, 2), (1, 2), (2, 2), (2, 3)],
                          [(1, 2), (2, 1), (2, 2), (3, 1)],
                          [(1, 1), (1, 2), (2, 1), (2, 2)],
                          [(1, 2), (2, 2), (2, 3), (3, 3)],
                          [(2, 2), (2, 3), (2, 4), (3, 2)],
                          [(1, 2), (2, 2), (2, 3), (2, 4)],
                          [(0, 2), (1, 2), (2, 1), (2, 2)],
                          [(1, 2), (2, 1), (2, 2), (3, 2)],
                          [(2, 1), (2, 2), (3, 2), (4, 2)],
                          [(1, 2), (1, 3), (2, 1), (2, 2)],
                          [(1, 2), (2, 1), (2, 2), (2, 3)],
                          [(2, 1), (2, 2), (2, 3), (3, 2)],
                          [(2, 1), (2, 2), (3, 2), (3, 3)],
                          [(1, 2), (2, 0), (2, 1), (2, 2)],
                          [(2, 0), (2, 1), (2, 2), (3, 2)],
                          [(2, 0), (2, 1), (2, 2), (2, 3)]]
        self.b_dict[5] = [[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
                          [(1, 2), (1, 3), (2, 2), (3, 2), (3, 3)],
                          [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2)],
                          [(1, 2), (2, 2), (2, 3), (2, 4), (3, 2)],
                          [(0, 2), (1, 2), (2, 2), (2, 3), (2, 4)],
                          [(1, 1), (1, 2), (2, 2), (3, 1), (3, 2)],
                          [(2, 1), (2, 2), (2, 3), (3, 1), (3, 3)],
                          [(1, 1), (1, 3), (2, 1), (2, 2), (2, 3)],
                          [(0, 2), (1, 2), (2, 1), (2, 2), (2, 3)],
                          [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)],
                          [(2, 1), (2, 2), (2, 3), (3, 2), (4, 2)],
                          [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],
                          [(1, 2), (2, 0), (2, 1), (2, 2), (3, 2)],
                          [(2, 0), (2, 1), (2, 2), (3, 2), (4, 2)],
                          [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]]

        self.lens = [1, 4, 8, 19, 15]
        self.indices = None
        self.block_array = np.zeros((5, 5), dtype=np.int)
        self.is_used = False

    '''
    1) 사용 가능한 블록임을 표시
    2) 블록의 길이를 먼저 sample
    3) 해당 길이의 블록의 종류 중 하나 sample
    4) 블록의 모양 표시
    '''
    def sample(self):
        self.is_used = False
        prob = self.lens / np.sum(self.lens)
        num_b = np.random.choice(5, p=prob) + 1
        samples = self.b_dict[num_b]
        k = np.random.randint(0, self.lens[num_b-1])
        self.indices = samples[k]
        for idx in self.indices:
            self.block_array[idx] += 1

    '''
    1) 블록의 모양 0으로 초기화
    2) 이미 사용된 블록임을 표시
    '''
    def used(self):
        self.block_array = np.zeros((5, 5), dtype=np.int)
        self.is_used = True
        self.indices = None


'''
Blockudoku 환경

blocks          : 3개의 Block object들을 담아두는 list
obs             : 현재 보드 상태와 주어진 블록 모양들을 저장해주는 dictionary 
                  보드 상태 <- obs['board']
                  블록 모양들 <- obs['blocks']
block_remain    : 사용하지 않은 블록 수
done            : 현재 Terminal state인지 여부
'''
class Blockudoku:
    def __init__(self):
        self.blocks = [Block(), Block(), Block()]
        self.obs = None
        self.block_remain = 0
        self.done = False
        self.completed_before = False

    '''
    환경 초기화
    '''
    def reset(self):
        self.done = False
        self.completed_before = False
        obs = dict()

        # board
        center = np.zeros((9, 9), dtype=np.int)
        left = np.ones((9, 2), dtype=np.int)
        right = np.ones((9, 2), dtype=np.int)
        top = np.ones((2, 13), dtype=np.int)
        bottom = np.ones((2, 13), dtype=np.int)
        board = np.concatenate((left, center, right), axis=1)
        board_array = np.concatenate((top, board, bottom), axis=0)
        obs['board'] = board_array
        obs['blocks'] = self.init_block()

        self.obs = obs

        return obs

    '''
    블록 초기화
    블록을 모두 사용하였을 때 or 환경을 초기화 할 때 사용
    '''
    def init_block(self):
        block_arrays = []
        for i in range(3):
            block = self.blocks[i]
            block.sample()
            block_arrays.append(block.block_array)
        self.block_remain = 3
        return block_arrays

    def render(self):
        print('board\n', self.obs['board'][2:11, 2:11], '\n')
        for k in range(3):
            print('block', k, '\n', self.obs['blocks'][k], '\n')

    '''
    transition

    1) 이미 사용된 블록을 선택한 경우, state 변화없이 0의 reward를 줌
    2) 블록을 불가능한 곳에 놓을 경우, state 변화없이 0의 reward를 줌
    3) 블록을 채우는데 성공하면, 채운 블록 수 만큼 reward를 줌
    4) 9칸을 채우면, 블록들이 지워지면서 18의 추가 reward를 주고,
    5) 한 번에 여러 번 지우면, 각 9만큼 보너스 reward를 줌
    6) 블록을 모두 사용했으면, 새로 블록들을 받음
    7) 주어진 블록들을 어느 곳에도 둘 수 없는 경우, terminal state로 판단
    '''
    def step(self, action):
        assert len(action) == 3, \
            'Action has 3 commands : (block number, row number, column number)'
        k, i ,j = action
        assert isinstance(k, int) and 0 <= k <= 2, \
            'block number is integer type number (0 or 1 or 2)'
        assert isinstance(i, int) and 2 <= i <= 10, \
            'row number should be in range [2, 10]'
        assert isinstance(j, int) and 2 <= j <= 10, \
            'column number should be in range [2, 10]'

        block = self.blocks[k]
        
        # 이미 사용된 블록 선택
        if block.is_used:
            return self.obs, 0, self.done

        # 불가능한 곳에 블록을 둠
        if not self.feasible_act(block.block_array, i, j):
            return self.obs, 0, self.done
        
        # 블록을 두는데 성공
        self.obs['board'][i-2:i+3, j-2:j+3] += block.block_array

        #############################################################
        ########## 블록을 지웠는 지 여부, 보너스 reward 코드 시작 ##########
        reward, is_completed = self.complete_test(block, i, j)
        if self.completed_before:
            reward += 9
        self.completed_before = is_completed
        ########################### 코드 끝 ##########################
        #############################################################

        # 이미 사용된 블록임을 표시
        block.used()
        self.obs['blocks'][k] = block.block_array
        self.blocks[k] = block
        self.block_remain -= 1

        # 블록을 모두 사용한 경우
        if self.block_remain == 0:
            self.obs['blocks'] = self.init_block()

        # terminal 여부
        done = self.done_test()

        return self.obs, reward, done

    '''
    블록을 둘 수 있는 곳인지 판단 : convolution layer 연산 활용
    '''
    def feasible_act(self, block_array, i, j):
        target_board = self.obs['board'][i-2:i+3, j-2:j+3]
        if np.sum(block_array*target_board) == 0:
            return True
        else:
            return False

    '''
    terminal 여부 판단 : convolution layer 연산 활용
    (cs231n의 im2col.py assignment 참고함)

    1) 보드 = 이미지, 블록 = 커널
    2) convolution 연산을 dot product로 한번에 하기 위해
    3) (5 x 5) 블록을 (25 x 1) column으로 변환
    4) 마찬가지로 (13 x 13) 보드를 convolution 연산에 맞게 (81 x 25) matrix로 변환
       (image size = 13, kernel size = 5, stride = 1 -> output size = (13 - 5)/1 + 1 = 9)
    5) 각 블록마다 convolution을 수행하고
    6) 0인 값(가능한 action)이 하나라도 있으면 non-terminal
    '''
    def done_test(self):
        done = True
        board = self.obs['board']

        # 행 번호
        i0 = np.repeat(np.arange(5), 5)
        i1 = np.repeat(np.arange(9), 9)
        i = i0.reshape(1, -1) + i1.reshape(-1, 1)

        # 열 번호
        j0 = np.tile(np.arange(5), 5)
        j1 = np.tile(np.arange(9), 9)
        j = j0.reshape(1, -1) + j1.reshape(-1, 1)

        # convolution 연산에 맞도록 indexing, reshaping
        board_col = board[i, j]

        # 블록마다 convolution 수행
        for k in range(3):
            if self.blocks[k].is_used:
                continue
            block_col = self.obs['blocks'][k].reshape(-1, 1)
            check = np.dot(board_col, block_col)
            if check.min() == 0:
                done = False
                break

        return done

    @staticmethod
    def get_indices(indices, i, j):
        # 블록이 채워지는 곳의 인덱스
        rst = []
        for idx in indices:
            row = idx[0] + i - 2
            col = idx[1] + j - 2
            rst.append((row, col))
        return rst

    @staticmethod
    def get_grid(idx):
        # 해당 인덱스에 해당하는 subgrid의 대표 인덱스
        row, col = idx
        g_r = (row - 2) // 3
        g_r = 3*g_r + 2
        g_c = (col - 2) // 3
        g_c = 3*g_c + 2
        return g_r, g_c

    def complete_test(self, block, i, j):
        reward = 0
        is_completed = False
        board = self.obs['board']
        indices = self.get_indices(block.indices, i, j)
        for idx in indices:
            bonus = 0
            row, col = idx
            g_r, g_c = self.get_grid(idx)
            if np.sum(self.obs['board'][row, 2:11]) == 9:
                board[row, 2:11] = 0
                bonus += 18
            if np.sum(self.obs['board'][2:11, col]) == 9:
                board[2:11, col] = 0
                bonus += 18
            if np.sum(self.obs['board'][g_r:g_r+3, g_c:g_c+3]) == 9:
                board[g_r:g_r+3, g_c:g_c+3] = 0
                bonus += 18
            if bonus == 0 and board[row, col] == 1:
                reward += 1
            else:
                reward += bonus
                is_completed = True
            self.obs['board'] = board

        return reward, is_completed


# test
if __name__ == '__main__':
    env = Blockudoku()
    obs = env.reset()

    while True:
        env.render()
        k = int(input('Select the block: '))
        i = int(input('row number: ')) + 1
        j = int(input('col number: ')) + 1
        act = (k, i, j)
        next_obs, reward, done = env.step(act)
        print('reward: ', reward)
        print('\n\n')
        if done:
            break

#%%