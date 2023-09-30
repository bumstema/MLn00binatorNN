from dataclasses import dataclass, field
import random 
from datetime import datetime, timedelta, time, date
from collections import namedtuple, deque
import hashlib
import torch
import cv2
import pytesseract
from PIL import Image, ImageFilter, ImageOps, ImageChops, GifImagePlugin
import numpy as np
from scipy.special import softmax
import math
import torchvision.transforms as transforms


from pyboy import PyBoy, WindowEvent
from pyboy.openai_gym import PyBoyGymEnv
from pyboy.plugins.base_plugin import PyBoyGameWrapper
from pyboy.core.mb import Motherboard

from ..model.q_agent import QLearningAgent
from ..data_io.utils import save_json_file, load_json_file, process_raw_frame
from ..data_io.utils import rgb2gray, save_gif_frames, unpack_nested_list, image_to_tensor
from ..data_io.utils import PlotGraph, progress
from ..data_io.const import DISPLAY, CURRENT_DIR_PATH, NN_CHECKPOINTS_PATH, NN_ROM_PATH, NN_TRAINING_PATH,WIDTH,HEIGHT
from .controls import BUTTONS, PyBoyGameControls

from .termination import LEVEL_TRUNCATION_STATES, LEVELS



        
        
# ----------------------------------- # -----------------------------------
@dataclass
class GameWrapperEnvironmentPokemon():
    cartridge_title = 'POKEMON YELLOW'
    
    # -----------------------------------
    def __init__(self, *args, **kwargs):

        self.player = 0  # Game Player Console (PyBoy)
        self.agent =  0  # Actor in the environment
        self.agent =  0  # Actor in the environment
        self.action = 0  # Button Input for game
        self.state = {}  # Result of Agent acting in the environment
        self.reward = 0  # Incentive towards certain actions
        self.done = False
        self.step = 0
        
        self.previous_states = {}
        self.current_state_hashes = []
        self.episode_cumulitive_reward = 0
        self.q_learning_stats = []
        
        self.state_gif_frames = []
        self.action_gif_frames = []
        self.acceleration_gif_frames = []

    # ----------------------------------- # -----------------------------------
    # ----------------------------------- # -----------------------------------
    def start_game(self, game_path, n_training_steps, train=True, emulation_speed=40, save_frames=False, save_gifs=False):
    
        with PyBoy(f'{game_path}', disable_renderer=False) as pyboy:
            print(pyboy.cartridge_title())
            self.player = pyboy
            self.player.set_emulation_speed(emulation_speed)
            
            self.agent = QLearningAgent()
            
            self.ep_min = 0
            self.ep_max = 1000
     
            self.fastest_run_n_steps = 1e9
            self.n_nonfailed_runs = self.ep_min
            
            self.highest_completed_level_label  = f'level-1'
            self.level_label = f'level-1'
            ep_load = self.ep_min
            try:
                model_name = f'Qlearning_model_({self.highest_completed_level_label}_ep_{ep_load}).pth'
                self.agent.load_model(NN_CHECKPOINTS_PATH+f'Qlearning_model_({self.highest_completed_level_label}_ep_{ep_load}).pth')
            except:
                print(f"Failed to load Qlearning model... ({model_name})")

            if train:
                try:
                    self.q_learning_stats = load_json_file( NN_CHECKPOINTS_PATH + f'Qlearning_stats_({self.level_label}).json' )
                except:
                    print(f"Starting new stats output file.")
                    self.q_learning_stats = []
            
            self.save_gifs = save_gifs
            self.full_training_gif_frames = []
            
            for episode in range(self.ep_min, self.ep_max):

                # Reset variables each new episode
                self.episode = episode
                self.state = []
                self.action = 0
                self.n_played_frames = 0
                self.episode_cumulitive_reward = 0
                self.previous_states = {}
                self.loss = 0
                self.q = 0
                self.start_time = datetime.now()
                self.state_gif_frames = []
                self.agent.nonrandom_decisions = 0

                # DEFINED end states for the actor to achieve (see:  termination.py)
                #---------------------------------------------------
                self.save_state_file = f'first_playable_state.state'
                #self.save_state_file = f'level-01-complete-(Exit_the_Bedroom).state'
                #self.save_state_file = f'level-2-complete-(Exit_the_House).state'
                #self.save_state_file = f'level-3-complete-(Get_To_Gary).state'
                
                self.load_game_state(file_obj= CURRENT_DIR_PATH + NN_ROM_PATH + self.save_state_file)
                self.state = self.update_state(self.action, colour_action_centered=False, grayscale=True)
                
                for index, step in enumerate(range(n_training_steps)):
                    self.step = step
                    self.agent.episode_steps = step
                    
                    #  - (1) - Agent takes action
                    self.action, self.reward =  self.agent.select_action(self.state)
                    self.episode_cumulitive_reward += self.reward
                    
                    #  - (2) - Pass action into environment
                    self.next_state = self.update_state(self.action, colour_action_centered=False, grayscale=True)
                    
                    #  - (3) - Cache system and response
                    self.agent.cache(self.state, self.action, self.reward, self.next_state, self.done)
   
                    self.failed = self.check_unacceptable_termination(self.next_state)
                    self.done = self.check_acceptable_termination(self.next_state)
    
                    #  - (4) - Apply updates to Q_learning model
                    if train:
                        _q, _loss = self.agent.learn()
                        if _q is not None: self.q, self.loss = _q, _loss
                    
                    #  - (5) - Advance states
                    self.state = self.next_state
                
                    if self.failed: break
                    if self.done: break
                
                newline = ''
                if self.done: newline = f'\n'
                prefix = f"Level:({self.level_label}) Episode:({episode + 1}) Failed:({self.failed})"
                suffix = f"|| steps: {step:^5}| states: {len(self.previous_states):^5}| nonrand: {self.agent.nonrandom_decisions:^5}|| q: {self.q:.3f} | reward: {self.reward:.3f} | T.reward: {self.episode_cumulitive_reward:.3f} ||{newline}"
                progress( self.episode, self.ep_max, prefix=f'{prefix}', suffix=f'{suffix}')

                if self.done:
                    self.n_nonfailed_runs += 1
                    self.agent.update_cache_for_fastest_runs()
                    if (self.step <= self.fastest_run_n_steps):
                        self.fastest_run_n_steps = self.step

                        print(f"Saving GameState....")
                        print(f"{prefix}")
                        print(f"{suffix}")
                        
                        #  - (6) - Log Parameters for Evaluation
                        self.log()
                    
                        # SAVE the next save state for the agent to repeat attempts of achieving.
                        #---------------------------------------------------------------------
                        #self.save_game_state( file_obj= CURRENT_DIR_PATH + NN_ROM_PATH + f'{self.level_label}-complete-(Exit_the_House).state')

                        #self.save_game_state( file_obj= CURRENT_DIR_PATH + NN_ROM_PATH + f'{self.level_label}-complete-(Get_To_Gary).state')
                        #print("............... ok!")

    # -----------------------------------
    # -----------------------------------
    def save_game_state(self, file_obj='file_obj.state'):
        self.player.save_state(open(file_obj, "wb"))

    # -----------------------------------
    def load_game_state(self, file_obj='file_obj.state'):
        self.player.load_state(open(file_obj, "rb"))

    # -----------------------------------
    def log(self):
    
        if self.step == 0:
            save_json_file({f'QNetworkDCN': str(self.agent.q_network)}, NN_CHECKPOINTS_PATH + f'QNetworkDCN_model.json' )
    
        if (self.done) :
            print(f"----------- Checkpoint: Saving Episode Stats...")
            self.q_learning_stats.append({'level':self.level_label,
                'episode':self.episode+1,
                'reward':self.reward,
                'total_reward':self.episode_cumulitive_reward,
                'loss':self.loss,
                'q':self.q,
                'steps':self.step,
                'n_played_frames':self.n_played_frames,
                'states_visited':len(self.previous_states),
                'nonrandom_decisions': self.agent.nonrandom_decisions,
                'done':self.done,
                'failed':self.failed,
                'time':str(datetime.now()-self.start_time)
            })
            save_json_file(self.q_learning_stats, NN_CHECKPOINTS_PATH + f'Qlearning_stats_({self.level_label}).json' )
            PlotGraph(self.q_learning_stats, filename=f'training_{self.level_label}.png')
            print(f"------------------ : Saving Model...\n")
            self.agent.save_model(NN_CHECKPOINTS_PATH+f'Qlearning_model_({self.level_label}_ep_{self.n_nonfailed_runs}).pth')

            if self.save_gifs :
                save_gif_frames( self.state_gif_frames , file_name=NN_TRAINING_PATH+f'RunTrajectory_({self.level_label}_ep_{self.n_nonfailed_runs}).gif')
                self.full_training_gif_frames.extend( self.state_gif_frames )
                save_gif_frames( self.full_training_gif_frames, file_name=NN_TRAINING_PATH+f'RunTrainingTrajectories_({self.level_label}_ep_{self.n_nonfailed_runs}).gif')

    # ----------------------------------- # -----------------------------------
    # PyBoy runs at 60 fps.
    # Take every 2nd tick to allow 30 fps.
    # Each input gives 3 frames with max 10 inputs per second.
    #
    # Skip ticks - [9,9] Clear one tile in movement between button presses
    #   (found in TAS guide that walk loop is 17 frames long)
    #   therefore, split between with 9 and 8 per press/release
    def press_key_return_frame(self, input_button):
        controller = PyBoyGameControls()
        support_mgr = self.player.botsupport_manager()
        screen = support_mgr.screen()

        consecutive_frames = []
        on_steps, off_steps  = random.choice([[8,9],[9,8]])
        
        consecutive_frames.append(screen.screen_ndarray())
        self.player.send_input(controller.press[input_button])
        _ = [self.player.tick() for fr in range(on_steps)]
    
        consecutive_frames.append(screen.screen_ndarray())
        self.player.send_input(controller.release[input_button])
        _ = [self.player.tick() for fr in range(off_steps)]
        
        consecutive_frames.append(screen.screen_ndarray())
        self.n_played_frames += 17
        return consecutive_frames
  
    # ----------------------------------- # -----------------------------------
    # ----------------------------------- # -----------------------------------
    def update_state(self, action_index, colour_action_centered=False, grayscale=False):
        state = []
        action = BUTTONS[action_index]
        state = self.press_key_return_frame(action)
        self.current_state_hashes = self.state_frame_hash(state, save_frame=False)

        if grayscale:
            state = self.combine_to_grayscale(state)
            if self.save_gifs:
                self.state_gif_frames.append( transforms.ToPILImage()(state) )
        return state

    # -----------------------------------
    def frame_hash(self, frame):
        return str(hashlib.sha256(np.ndarray.flatten(frame)).hexdigest())
    
     # -----------------------------------
    def state_frame_hash(self, state, save_frame=False, save_gif=False):
        full_set_of_hashes = []
        imgs_g = [ rgb2gray(state[subframe]) for subframe in range(3)]
        imgs = np.ones( np.array(state[0]).shape )
        imgs[:,:,0] = imgs_g[0]
        imgs[:,:,1] = imgs_g[1]
        imgs[:,:,2] = imgs_g[2]
        imgs_g.append(imgs)
        for frame in imgs_g:
            frame_hash_label = str(hashlib.sha256(np.ndarray.flatten(frame)).hexdigest())
            if frame_hash_label not in self.previous_states:
                self.previous_states.update({frame_hash_label:self.step})
                if save_frame:
                    self.save_frame(frame, frame_hash_label)
            full_set_of_hashes.append(frame_hash_label)
        return full_set_of_hashes

   
    # -----------------------------------
    def save_frame(self, frame, frame_label):
        with Image.fromarray((frame).astype(np.uint8)) as pic:
            pic.convert("RGB").save(NN_TRAINING_PATH+f'{frame_label}.png')


    # -----------------------------------
    def combine_to_grayscale(self, state):
        imgs_g = [ rgb2gray(state[subframe]) for subframe in range(3)]
        imgs = np.ones( np.array(state[0]).shape )
        imgs[:,:,0] = imgs_g[0]
        imgs[:,:,1] = imgs_g[1]
        imgs[:,:,2] = imgs_g[2]
        return image_to_tensor(imgs)
 
    # -----------------------------------
    def normalize_pixels_around_zero(self, img):
        mean_val = np.mean(img)
        std_val = np.std(img)
        norm_frame = img - mean_val
        if std_val == 0 : std_val = 1
        norm_frame = norm_frame / std_val
        return norm_frame.astype(np.float32)
    
    # -----------------------------------
    def normalize_pixels_to_mid_RGB(self, img):
        mean_val = np.mean(img)
        std_val = np.std(img)
        diff_frame = img - mean_val
        max_pixel_val = np.max(diff_frame)
        if max_pixel_val == 0 : max_pixel_val = 1
        diff_frame = ((diff_frame / (max_pixel_val)) * 255//2) + (255//2)
        return diff_frame.astype(np.uint8)
        
    # -----------------------------------
    def framestep_image_difference(self, previous_frame, current_frame):
        diff_frame = (current_frame - previous_frame)
        return self.normalize_pixels_around_zero(diff_frame)

    # -----------------------------------
    def framestep_velocity_estimation(self, previous_frame, current_frame, future_frame, x_hat=1):
        x = [previous_frame, current_frame, future_frame]
        dxdt = (-  x_hat )*(( x[2] - (4*x[1]) + (3*x[0]) ) //2 )
        return self.normalize_pixels_around_zero(dxdt)

    # -----------------------------------
    def framestep_acceleration_estimation(self, previous_frame, current_frame, future_frame):
        x = [previous_frame, current_frame, future_frame]
        d2xdt2 = (x[0] + x[2] - (2*x[1]))
        return self.normalize_pixels_around_zero(d2xdt2)

    # -----------------------------------
    def build_state_encoded_images(self, state):
        R,G,B = 0,1,2
        
        ######
        # Turn the states' 3 screen update frames into: #
        # 1. State encoding as RGB
        gray_frames = [ rgb2gray(subframe) for subframe in state]
        state_encoding = np.ones(state[0].shape)
        state_encoding[:,:,R] = gray_frames[0]
        state_encoding[:,:,G] = gray_frames[1]
        state_encoding[:,:,B] = gray_frames[2]

        ######
        # 2. Action map
        action_map = np.ones(state[0].shape)
        back_diff = self.framestep_image_difference(gray_frames[0], gray_frames[1] )
        center_diff = self.framestep_image_difference(gray_frames[0], gray_frames[2])
        forward_diff = self.framestep_image_difference(gray_frames[1], gray_frames[2])
        action_map[:,:,R] = back_diff
        action_map[:,:,G] = center_diff
        action_map[:,:,B] = forward_diff
        

        ######
        # 3. Acceleration map
        acceleration_map = np.ones(state[0].shape)
        back_diff = self.framestep_image_difference(action_map[:,:,R], action_map[:,:,G])
        center_diff = self.framestep_acceleration_estimation(gray_frames[0],gray_frames[1],gray_frames[1])
        forward_diff = self.framestep_image_difference(action_map[:,:,G], action_map[:,:,B])
        acceleration_map[:,:,R] = self.normalize_pixels_around_zero(back_diff)
        acceleration_map[:,:,G] = center_diff
        acceleration_map[:,:,B] = self.normalize_pixels_around_zero(forward_diff)
        
        if self.save_gifs:
            _ = [ self.save_frame(obj, f'{idx}_'+self.frame_hash(obj) )  for idx, obj in enumerate([state_encoding, action_map, acceleration_map]) ]
            self.state_gif_frames.append(Image.fromarray(state_encoding.astype(np.uint8)))
            self.action_gif_frames.append(Image.fromarray(action_map.astype(np.uint8) ))
            self.acceleration_gif_frames.append(Image.fromarray(acceleration_map.astype(np.uint8) ))

        torch_ready_images = [np.transpose(img, (2, 0, 1)) for img in (state_encoding, action_map, acceleration_map)]
        
        return tuple(torch_ready_images)

    # -----------------------------------
    def check_unacceptable_termination(self, state):
        current_level = LEVELS.index(self.level_label)
        states_to_check = self.current_state_hashes
        
        stop_state = False
        bonus_reward = 2
        
        # Allow the agent to move off the previous end state
        if self.step < 10: return False

        #any(item in states_to_check for item in all states below current TRUNCATION_STATES_LEVEL)
        if any(item in states_to_check for item in sum(LEVEL_TRUNCATION_STATES[:current_level],[]) ) | (self.step > self.fastest_run_n_steps):
            stop_state = True
            
            # Adds NEGATIVE bonus reward to the last 8 movements
            self.reward += - bonus_reward

            n_last_actions = len(self.agent.memory) - len(self.agent.accepted_memory)
            n_last_actions = n_last_actions // 10
            for index, idx in enumerate(range(-n_last_actions,0)):
                saved_reward = self.agent.memory[idx]['reward']
                self.agent.memory[idx]['reward'] = saved_reward - ((index+1)/n_last_actions)*bonus_reward
            
        return stop_state
        
    # -----------------------------------
    def check_acceptable_termination(self, state):
        current_level = LEVELS.index(self.level_label)
        states_to_check = self.current_state_hashes
        
        stop_state = False
        bonus_reward = 2
        
        # Allow the agent to move off the previous end state
        if self.step < 10: return False
        
        if any(item in states_to_check for item in LEVEL_TRUNCATION_STATES[current_level]):
            stop_state = True

            # Adds POSITIVE bonus reward to the last 8 movements
            self.reward += bonus_reward

            n_last_actions = len(self.agent.memory) - len(self.agent.accepted_memory)
            n_last_actions = n_last_actions // 10
            for index, idx in enumerate(range(-n_last_actions,0)):
                saved_reward = self.agent.memory[idx]['reward']
                self.agent.memory[idx]['reward'] = saved_reward + ((index+1)/n_last_actions)*bonus_reward
            
        return stop_state

    # -----------------------------------
    # Works only with tick rate set at press:4, release:2
    def get_to_first_playable_state(self, step):
        
        self.action = 'a'
        # Name Choice "Yellow"
        if step == 563:
            self.action = 'down'
        #  Rival Name Choice "Gary"
        if step == 709:
            self.action =  'down'
        if step == 710:
            self.action =  'down'
        if step == 711:
            self.action =  'down'
        
        # Save Game at First Movement
        if 954 > step > 951:
            self.action =  'start'
        if  965 > step > 953:
            self.action =  'down'

        #
        if 1099 == step:
            self.save_game_state(file_obj= CURRENT_DIR_PATH + NN_ROM_PATH + f'first_playable_state.state')
        if 1107 == step:
            self.load_game_state(file_obj= CURRENT_DIR_PATH + NN_ROM_PATH + f'first_playable_state.state')

    # -----------------------------------
    def detect_text_in_frame(self, frame):
        cv_frame = cv2.cvtColor(np.array(frame,dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        cv_frame = cv2.resize(cv_frame, TEXT_DETECT_DISPLAY)
        text = pytesseract.image_to_string(cv_frame)
        return text






"""


self.last_actions.insert(0,self.action)
if len(self.last_actions) > max_buffer_size: self.last_actions.pop(-1)
(self.reward, episode_complete)  = self.select_reward(self.next_state)
self.next_state.reward = self.reward
self.episode_cumulitive_reward += self.reward
self.last_rewards.insert(0, self.reward)
if len(self.last_rewards) > max_buffer_size: self.last_rewards.pop(-1)
self.last_states.insert(0, self.next_state)
if len(self.last_states) > max_buffer_size: self.last_states.pop(-1)

self.previous_states.setdefault(self.next_state.frame_hash, self.next_state)

self.previous_states[self.next_state.frame_hash].n_visited += 1

if self.next_state.frame_hash != self.state.frame_hash:

    n_played_frames += 1
    print(f"{step:<6} {BUTTONS[self.action]:^8} {self.reward:^4} \t{self.next_state.frame_hash}  {self.episode_cumulitive_reward:^6} {self.next_state.text}")
    
    last_buffer = self.frame_buffer
    self.frame_buffer.insert(0, np_grayscale(self.next_state.previous_frame))
    self.frame_buffer.insert(0, np_grayscale(self.next_state.difference_frame))
    self.frame_buffer.insert(0, np_grayscale(self.next_state.frame))
    self.frame_buffer.pop(-1)
    self.frame_buffer.pop(-1)
    self.frame_buffer.pop(-1)
    
    self.last_frame_buffers.insert(0, self.frame_buffer)
    if len(self.last_frame_buffers) > max_buffer_size: self.last_frame_buffers.pop(-1)
    
    
    #self.agent.train(np.array([last_buffer]), self.action, self.reward, np.array([self.frame_buffer]))

# remove buttons: a/b/select/start
#self.action = (self.action % 4)
# Only down and left/right
#self.action = random.choice([1,2,3])
                    
                    
    
    
"""
        #truncate_states_level_2 = [f'35cbb99dbca4ad6e69d781f266613094103097152632cfbf13f908bc422ddad2',f'd0aba30cd127e8165e9d84e68cc80005cd5f42f74eee9e4765b3d900992857c2',f'87b7409019d1fe7b8757bab8ccf1c312476e6370e516e2c9d79824b5ff4d0959',f'82cfbe535bfc6829bdd27f31f0900995c6a0b0d475cdbb55042d08f493d56705',f'38307ce7c68b80ebb6dc0a6fe69264f663fa4eb16fad892a302c03e8bd1e15cd',f'5feac59f521debe6ca5dc3130fed3d1c8b4966ae65d972363d10424cebe54b90',f'e0110a88046b5eff4eb79966b162848fb0ef51e852d6188167982b361456d28c']
        
        #(state.previous_frame, state.tween_frame , state.frame) = self.press_key_return_frame(state.action)
        #gray_frame = cv2.cvtColor(np.array(state.frame, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        #gray_prev_frame = cv2.cvtColor(np.array(state.previous_frame,dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        #state.frame_hash = str(hashlib.sha256(np.ndarray.flatten(gray_frame)).hexdigest())
        #state.previous_state = str(hashlib.sha256(np.ndarray.flatten(gray_prev_frame)).hexdigest())
        #state.text = self.detect_text_in_frame(state.frame).replace('\n', ' ').strip()
        #(state.difference_frame, state.total_different_pixels)  = self.framestep_image_difference(state.previous_frame, state.frame)
    
