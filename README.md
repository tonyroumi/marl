How to run : 
python -m marl.train 

if hp something is not installed, install it via pip (pip install x)

adjust the iteration number at basic__marl_agent.py 
change : 
    def learn(self, total_iterations: int = 10) -> None:
        """
        Main training loop for the PPO agent.

        Args:
            total_iterations (int): Number of learning iterations to perform.
        """
        obs, info = self.env.reset()
        actor_obs, critic_obs = self.process_observations(obs)
        self.train_mode() 
        
 this from 10 to whatever number you want
