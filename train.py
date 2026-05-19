from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from pvz_env import PvZEnv

def main():
    print("Loading the PvZ Environment...")
    env = PvZEnv()
    
    # 1. THE SAFETY CHECK
    # This built-in Gym function plays a few random frames to ensure your
    # observation and action spaces perfectly match the arrays you output.
    check_env(env)
    print("Environment is valid! No crashes detected.")

    # 2. INITIALIZE THE BRAIN (PPO)
    # MultiInputPolicy: Tells the AI to expect a Dictionary of different data shapes
    # env: The game we just built
    # verbose=1: Prints out statistics to the terminal so you can watch it learn
    # tensorboard_log: Saves graphs of the AI's progress
    print("Initializing PPO Neural Network...")
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./ppo_pvz_tensorboard/"
    )

    # 3. TRAIN THE AI
    # total_timesteps is how many 'Ticks' the AI is allowed to play.
    # 100,000 is a good starting point to see if it learns to place Sunflowers.
    print("Starting Training Loop...")
    model.learn(total_timesteps=100000)

    # 4. SAVE THE BRAIN
    print("Training Complete! Saving model...")
    model.save("pvz_ai_model_SunflowersLockFix")

if __name__ == "__main__":
    main()