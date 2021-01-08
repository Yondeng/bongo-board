from gym.envs.registration import register

register(
    id='BongoBoard-v0',
    entry_point='bongo_board.bongoboard:bongo_board'
)

register(
    id='pole_kinetic-v0',
    entry_point='bongo_board.pole:long_pole'
)