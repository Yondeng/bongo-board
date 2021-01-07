from gym.envs.registration import register

register(
    id='bongo-board-v0',
    entry_point='bongo-board.bongo-board:bongo_board'
)