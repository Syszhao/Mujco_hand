"""
MuJoCo渲染+模型测试脚本
1. 先用方法一（环境render）测试渲染
2. 渲染窗口关闭后自动加载模型并进行测试
"""

import os
import time
import mujoco
import numpy as np
from train_rotate_balls import RotateBallsEnv
from stable_baselines3 import PPO
import argparse
import sys

# 检查MuJoCo版本
print(f"MuJoCo版本: {mujoco.__version__}")

def test_render_method1():
    """测试方法1: 使用环境的render方法"""
    print("\n=== 测试方法1: 使用环境的render方法 ===")
    try:
        env = RotateBallsEnv(render_mode="human")
        print("环境创建成功")
        
        # 重置环境
        env.reset()
        print("环境重置成功")
        
        # 尝试渲染
        print("尝试渲染... (关闭窗口后继续)")
        for i in range(100):
            action = env.action_space.sample()
            env.step(action)
            env.render()
            time.sleep(0.05)
        print("方法1测试完成")
        env.close()
        return True
    except Exception as e:
        print(f"方法1测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_trained_policy(model_path, max_episodes=30, max_steps=5000, render=True, deterministic=True):
    """加载训练好的模型并进行测试"""
    print(f"\n=== 加载模型并测试: {model_path} ===")
    try:
        env = RotateBallsEnv(render_mode="human" if render else None)
        model = PPO.load(model_path)
        print("模型加载成功")
        for ep in range(max_episodes):
            obs, _ = env.reset()
            total_reward = 0
            for step in range(max_steps):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if render:
                    env.render()
                if terminated or truncated:
                    print(f"回合{ep+1}提前终止，步数: {step+1}, 累计奖励: {total_reward:.2f}")
                    break
            print(f"回合{ep+1}结束，总奖励: {total_reward:.2f}")
        env.close()
    except Exception as e:
        print(f"模型测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo渲染+模型测试")
    parser.add_argument('--skip-test', action='store_true', help='只渲染不进行模型测试')
    parser.add_argument('--model', type=str, default="./logs/final_model", help='模型路径')
    args = parser.parse_args()

    # 只用方法一渲染
    ok = test_render_method1()
    if ok and not args.skip_test:
        test_model_with_trained_policy(args.model)
    else:
        print("渲染失败或已选择只渲染，不进行模型测试。") 