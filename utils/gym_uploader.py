import gym
import secrets

def upload_results(folder):
    gym.upload(folder, api_key=secrets.api_key)
