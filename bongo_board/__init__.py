from gym.envs.registration import register

register(
    id='BongoBoard-v0',
    entry_point='bongo-board.bongo-board:bongo_board'
)