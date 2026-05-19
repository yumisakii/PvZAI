import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PvZEnv(gym.Env):
    def __init__(self):
        super(PvZEnv, self).__init__()
        
        # --- THE RULES ---
        
        # Action Space: [Action_Type, Plant_ID, Row, Column]
        # Action_Type: 0 (Wait), 1 (Plant)
        # Plant_ID: 0 (None), 1 (Sunflower), 2 (Peashooter)
        # Row: 0 to 4 (5 total)
        # Column: 0 to 8 (9 total)
        self.action_space = spaces.MultiDiscrete([2, 3, 5, 9])
        
        # Observation Space: A Dictionary containing the 1D Board and the Sun Count
        self.observation_space = spaces.Dict({
            # The FLATTENED Grid: 5 * 9 * 3 = 135.
            "board": spaces.Box(low=0, high=10, shape=(135,), dtype=np.int32),
            # The Bank: A single number for Sun
            "sun": spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32)
        })

        self.tick_counter = 0

        self.lives = 4

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Create the empty 3D lawn
        self.board_state = np.zeros((5, 9, 3), dtype=np.int32)
        
        # Give the AI starting Sun
        self.sun_count = np.array([150], dtype=np.int32)
        
        self.tick_counter = 0
        self.lives = 4

        # Combine them into a dictionary to send to the AI
        observation = {
            "board": self.board_state.flatten(),
            "sun": self.sun_count
        }
        
        return observation, {}

    def step(self, action):
        action_type = action[0]
        plant_id = action[1]
        row = action[2]
        col = action[3]
        
        prices = {1: 50, 2: 100}
        
        reward = 0.01 # Base Survival Reward
        terminated = False # This tells Gymnasium if the game is over

        # AI decision making
        if action_type == 1:
            # Check if we have enough sun AND the tile is empty AND the plant_id is valid
            if plant_id in prices and self.sun_count[0] >= prices[plant_id] and self.board_state[row, col, 0] == 0:
                self.sun_count[0] -= prices[plant_id]     # Pay the cost
                self.board_state[row, col, 0] = plant_id  # Place the plant
                if plant_id == 1:
                    reward += 0.02 # Reward for successfully planting a Sunflower
                elif plant_id == 2:
                    reward += 0.06 # Reward for successfully planting a Peashooter

        elif action_type == 0:
            pass # The AI chose to wait. Do nothing this tick.
        
        # Zombie Movement
        self.tick_counter += 1
        
        if self.tick_counter % 10 == 0:
            for i in range(5):
                for j in range(9):
                    # Check how many zombies are on this specific tile
                    zombies_here = self.board_state[i, j, 1]
                    plant_here = self.board_state[i, j, 0]
                    sunflower_here = self.board_state[i, j, 0] == 1
                    
                    # Plant is eaten here
                    if plant_here > 0 and zombies_here > 0:
                        self.board_state[i, j, 0] = 0
                        reward -= 0.1
                        continue # If the plant is eaten, zombies don't move this tick

                    #Zombies move here
                    if zombies_here > 0:
                        if j == 0:
                            self.lives -= zombies_here
                            reward -= 1.0
                            self.board_state[i, j, 1] -= zombies_here
                            if self.lives <= 0:
                                terminated = True
                        else:
                            self.board_state[i, j, 1] -= zombies_here
                            self.board_state[i, j-1, 1] += zombies_here
                    
                    # Sunflower generates sun here
                    if sunflower_here:
                        self.sun_count[0] += 25

        # Zombie Spawning
        if self.tick_counter % 35 == 0:
            spawn_row = np.random.randint(0, 5)
            self.board_state[spawn_row, 8, 1] += 1

        # Peashooter Attacks
        if self.tick_counter % 15 == 0:
            for i in range(5):
                for j in range(9):
                    if self.board_state[i, j, 0] == 2:
                        for scan_col in range(j, 9):
                            if self.board_state[i, scan_col, 1] > 0: # Zombie dies here
                                self.board_state[i, scan_col, 1] -= 1
                                reward += 0.3
                                break
        
        observation = {"board": self.board_state.flatten(), "sun": self.sun_count}
        info = {} # We can leave this empty for now
        truncated = False # We use this if there is a strict time limit (e.g. 5 minutes)
        
        return observation, reward, terminated, truncated, info