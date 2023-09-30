from pyboy import PyBoy, WindowEvent
from dataclasses import dataclass, field


#------------------------------------------------------------
BUTTONS = ['up','down','right','left','a','b','select','start']
#------------------------------------------------------------
@dataclass
class PyBoyGameControls():
    def __init__(self,):
        self.press = {'up':WindowEvent.PRESS_ARROW_UP,
            'down':WindowEvent.PRESS_ARROW_DOWN,
            'right':WindowEvent.PRESS_ARROW_RIGHT,
            'left':WindowEvent.PRESS_ARROW_LEFT,
            'a':WindowEvent.PRESS_BUTTON_A,
            'b':WindowEvent.PRESS_BUTTON_B,
            'select':WindowEvent.PRESS_BUTTON_SELECT,
            'start':WindowEvent.PRESS_BUTTON_START}
 
        self.release = {'up':WindowEvent.RELEASE_ARROW_UP,
            'down':WindowEvent.RELEASE_ARROW_DOWN,
            'right':WindowEvent.RELEASE_ARROW_RIGHT,
            'left':WindowEvent.RELEASE_ARROW_LEFT,
            'a':WindowEvent.RELEASE_BUTTON_A,
            'b':WindowEvent.RELEASE_BUTTON_B,
            'select':WindowEvent.RELEASE_BUTTON_SELECT,
            'start':WindowEvent.RELEASE_BUTTON_START}
#------------------------------------------------------------


