import os
import sys
from stable_baselines3 import SAC
import gymnasium as gym
import highway_env
import torch
import imageio
import cv2
from gymnasium.envs.registration import register

def save_best_episode_as_media(best_images, output_dir, slow_factor=40):
    if not best_images:
        print("저장할 이미지가 없습니다.")
        return
    
    gif_path = os.path.join(output_dir, 'best_parking_simulation.gif')
    gif_images = [img for img in best_images for _ in range(slow_factor)]
    imageio.mimsave(gif_path, gif_images, fps=240)
    print(f"최고 보상 에피소드 GIF가 {gif_path}에 저장되었습니다.")
    
    height, width, layers = best_images[0].shape
    mp4_path = os.path.join(output_dir, 'best_parking_simulation.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(mp4_path, fourcc, 240, (width, height))

    for img in best_images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for _ in range(slow_factor):
            video.write(img_bgr)

    video.release()
    print(f"에피소드가 MP4로 {mp4_path}에 저장되었습니다.")

def evaluate(env, device, start_positions, goal_positions):
    model = SAC.load('MVP_AutoParkingModel', env=env, device=device)
    best_episode_reward = -float('inf')
    best_images = []

    for episode in range(10):
        obs, _ = env.reset(start_positions=start_positions, goal_positions=goal_positions)
        episode_reward = 0
        images = []

        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            img = env.render()
            if img is not None:
                images.append(img)
            if done or info.get("is_success", False):
                print(f"에피소드 {episode + 1} - reward: {episode_reward}, success? {info.get('is_success', False)}")
                break

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_images = images

    os.makedirs(output_dir, exist_ok=True)
    save_best_episode_as_media(best_images, output_dir)

def get_position_input(prompt):
    while True:
        position = input(prompt)
        if position.isdigit():
            position = int(position)
            if 0 < position < 9:
                return position
            else:
                print(f"잘못된 위치: {position}. 1에서 8 사이의 숫자를 입력하세요.")
        else:
            print(f"잘못된 위치: {position}. 정수를 입력하세요.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA를 사용할 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    print(f"사용 중인 장치: {device}")

    register(
        id='CustomParkingEnv_Kimura_v100',
        entry_point='env_final:ParkingEnv',
        max_episode_steps=1000,
    )

    env = gym.make("CustomParkingEnv_Kimura_v100", render_mode="rgb_array")
    output_dir = "output"

    start_position = get_position_input("차량의 시작 위치를 입력하세요: ")
    while True:
        goal_position = get_position_input("차량의 목표 위치를 입력하세요: ")
        if start_position != goal_position:
            break
        print("시작 위치와 목표 위치가 같습니다. 다시 입력하세요.")

    evaluate(env, device, [start_position], [goal_position])
