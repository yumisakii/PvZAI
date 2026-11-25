from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from pvz_env import PvZEnv

def main():
    print("Loading the PvZ Environment...")
    env = PvZEnv()
    check_env(env)

    print("Environment is valid! No crashes detected.")
    print("Initializing PPO Neural Network...")

    # From scratch:
    #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_pvz_tensorboard/")

    # Coninue training:
    model = PPO.load("pvz_ai_model_PlantingZones", env=env, tensorboard_log="./ppo_pvz_tensorboard/")

    print("Starting Training Loop...")
    model.learn(total_timesteps=100000)

    print("Training Complete! Saving model...")
    model.save("pvz_ai_model_PlantingZones")

if __name__ == "__main__":
    main()