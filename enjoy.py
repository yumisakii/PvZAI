from stable_baselines3 import PPO
from pvz_env import PvZEnv
import time
import os

def print_board(obs, tick):
    # Clear the terminal so it looks like an animation
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # THIS IS THE MAGIC LINE: Fold the 1D array back into a 3D box
    board = obs["board"].reshape((5, 9, 3))
    sun = obs["sun"][0]
    
    print(f"=== TICK: {tick} | SUN: {sun} ===")
    
    for row in range(5):
        row_string = ""
        for col in range(9):
            zombies = board[row, col, 1]
            plant_id = board[row, col, 0]
            
            if zombies > 0:
                row_string += f"[ Z{zombies} ]" # Shows Z and how many are stacked
            elif plant_id == 1:
                row_string += "[ S  ]"
            elif plant_id == 2:
                row_string += "[ P  ]"
            else:
                row_string += "[ .  ]"
        print(row_string)
    print("===============================\n")

def main():
    print("Loading the environment and AI...")
    env = PvZEnv()
    
    # Load your trained brain
    model = PPO.load("pvz_ai_model_SunflowersLockFix")

    obs, info = env.reset()
    
    # Watch it play for 500 ticks
    for i in range(500):
        # 1. ASK THE AI FOR A MOVE
        action, _states = model.predict(obs)
        
        # 2. STEP THE GAME FORWARD
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 3. DRAW THE BOARD
        print_board(obs, env.tick_counter)
        
        # 4. SLOW IT DOWN (So you can watch it at roughly 10 Ticks per second)
        time.sleep(0.1)
        
        if terminated:
            print(f"GAME OVER! A Zombie ate the brain at Tick {env.tick_counter}.")
            time.sleep(3) # Pause so you can see the failure
            obs, info = env.reset()

if __name__ == "__main__":
    main()