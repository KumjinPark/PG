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
        self.b_dict[5] = [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
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
                          [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]]

        self.lens = [1, 4, 8, 19, 15]
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
        num_b = np.random.randint(1, 6)
        samples = self.b_dict[num_b]
        k = np.random.randint(0, self.lens[num_b-1])
        indices = samples[k]
        for idx in indices:
            self.block_array[idx] += 1

    '''
    1) 블록의 모양 0으로 초기화
    2) 이미 사용된 블록임을 표시
    '''
    def used(self):
        self.block_array = np.zeros((5, 5), dtype=np.int)
        self.is_used = True


'''
Blockudoku 환경

blocks  : 3개의 Block object들을 담아두는 list
obs     : 현재 보드 상태와 주어진 블록 모양들을 저장해주는 dictionary 
          보드 상태 <- obs['board']
          블록 모양들 <- obs['blocks']  
'''
class Blockudoku:
    def __init__(self):
        self.blocks = [Block(), Block(), Block()]
        self.obs = None

    '''
    환경 초기화
    '''
    def reset(self):
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

        # blocks
        block_arrays = []
        for i in range(3):
            block = self.blocks[i]
            block.sample()
            block_arrays.append(block.block_array)
        obs['blocks'] = block_arrays

        self.obs = obs

        return obs

    '''
    transition
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

        if block.is_used:
            return self.obs

        target_board = self.obs['board'][i-2:i+3, j-2:j+3]
        if np.sum(block.block_array*target_board) > 0:
            return self.obs

        self.obs['board'][i-2:i+3, j-2:j+3] += block.block_array
        block.used()
        self.obs['blocks'][k] = block.block_array
        self.blocks[k] = block
        return self.obs


# test
if __name__ == '__main__':
    env = Blockudoku()
    obs = env.reset()
    print('board\n', obs['board'], '\n')
    for k in range(3):
        print('block', k, '\n', obs['blocks'][k], '\n')

    # action test
    next_obs = env.step((0, 2, 2))
    print('board\n', next_obs['board'], '\n')
    for k in range(3):
        print('block', k, '\n', next_obs['blocks'][k], '\n')
