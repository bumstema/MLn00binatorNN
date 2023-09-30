from MLn00binatorNN.data_io.const import GAME_PATH
from MLn00binatorNN.environment.env import GameWrapperEnvironmentPokemon

#---------------------------------------------------
#---------------------------------------------------
def main():

    environment = GameWrapperEnvironmentPokemon()
    
    environment.start_game(f'{GAME_PATH}', 40000, train=True, save_frames=False)
    
# -------  --------
if __name__ == '__main__':
    main()

