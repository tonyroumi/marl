from robosuite.wrappers.wrapper import Wrapper

class CustomRewardWrapper(Wrapper):
    """ Custom reward wrapper """
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, action=None):
        return 0 #example reward

        
        
        