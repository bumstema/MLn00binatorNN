import sys, os, os.path

WIDTH = 160
HEIGHT = 144
DISPLAY = (HEIGHT, WIDTH)
COLOUR_DISPLAY = (HEIGHT, WIDTH, 3)
TEXT_DETECT_DISPLAY = (2*HEIGHT, 2*WIDTH)


#  ROM file downloaded from: https://pokemonrom.net/

CURRENT_DIR_PATH = os.getcwd()
NN_ROM_PATH = '/game_data/pokemon/emulator_ROMs/'
PKMN_FILE_NAME = 'Pokemon - Yellow Version - Special Pikachu Edition (USA, Europe) (CGB+SGB Enhanced).gb'
GAME_PATH = CURRENT_DIR_PATH + NN_ROM_PATH + PKMN_FILE_NAME
TRAINING_SET_PATH = '/game_data/pokemon/training_set/'
NN_FRAMES_PATH = 'state_frames/'
NN_TRAINING_PATH =  CURRENT_DIR_PATH + TRAINING_SET_PATH + NN_FRAMES_PATH
TRAINING_VIDEO_NANE = 'Pokemon Yellow Any% Glitchless Speedrun - 1_53_37 (Former World Record)-zo_mODRQfXo.mp4'

NN_CHECKPOINTS_PATH = CURRENT_DIR_PATH + TRAINING_SET_PATH + f'checkpoints/'
