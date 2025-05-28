import numpy as np
import gymnasium as gym  # 使用 Gymnasium 而不是 Gym
from gymnasium import spaces
from gymnasium.utils import seeding
import os
from typing import Optional, Dict, Any

import mujoco
from mujoco import MjModel, MjData, viewer as mj_viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize


class HandEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, xml_file='rm_75_6f_description.xml', render_mode: Optional[str] = None):
        super(HandEnv, self).__init__()

        full_path = os.path.abspath(os.path.join(os.getcwd(), xml_file))
        print(f"Loading model from {full_path}")

        if not os.path.exists(full_path):
            raise IOError(f"File {full_path} does not exist")
        if not full_path.startswith(os.getcwd()):
            raise ValueError("XML file path is not within current working directory")

        # 加载模型
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)

        # 获取目标物体 body id
        body_name = "cube_body1"
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.cube_body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in the XML model")

        num_actuators = self.model.nu

        # 自动计算 observation 的维度
        obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float64
        )

        # 动作空间
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_actuators,),
            dtype=np.float64
        )

        # 初始化 viewer
        self.render_mode = render_mode
        self.viewer = None

        # 随机种子
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_action(self, action):
        self.data.ctrl[:] = action

    def _get_observation(self):
        cube_pos = self.data.xpos[self.cube_body_id]
        cube_quat = self.data.xquat[self.cube_body_id].copy()
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            cube_pos.copy(),
            cube_quat.copy()
        ])

    def _compute_reward(self) -> float:
        target_quat = np.array([1., 0., 0., 0.])  # 目标四元数
        quat_diff = self.data.xquat[self.cube_body_id]
        reward_orientation = -np.linalg.norm(quat_diff - target_quat)

        action_cost = -0.01 * np.linalg.norm(self.data.ctrl)

        return reward_orientation + action_cost

    def _is_done(self) -> bool:
        return self.data.time > 2.0

    def step(self, action):
        self._set_action(action)
        mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = False  # 可根据最大步数设置为 True
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # 添加随机扰动
        self.data.qpos[:] += self.np_random.uniform(-0.1, 0.1, size=self.data.qpos.shape)
        self.data.qvel[:] += self.np_random.uniform(-0.1, 0.1, size=self.data.qvel.shape)

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        info = {}
        return observation, info

    def render(self, mode='human', width=500, height=500):
        if self.render_mode == "human" and self.viewer is None:
            try:
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"Failed to launch viewer: {e}")
        elif self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# 创建环境函数
def make_env():
    def _thunk():
        env = HandEnv(xml_file='rm_75_6f_description.xml', render_mode='human')
        return env

    return _thunk


if __name__ == '__main__':
    # 设置使用的环境数量（即使用多少个CPU核心）
    num_envs = 1

    # 创建向量化环境用于训练
    train_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 创建单个环境用于测试和渲染
    test_env = DummyVecEnv([make_env()])
    test_env = VecNormalize.load("vec_normalize.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False

    # 创建 PPO 模型
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        batch_size=64,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),  # Updated for SB3 v1.8.0+
        tensorboard_log="./ppo_hand_tensorboard/",
        device="cpu"  # 强制使用 CPU
    )

    # 开始训练
    # model.learn(total_timesteps=1000000)  # 增加训练时间

    # 保存模型和 VecNormalize 参数
    # model.save("ppo_hand_env_normalized")
    # train_env.save("vec_normalize.pkl")
    model=PPO.load("ppo_hand_env_normalized")

    # 测试训练好的策略
    obs = test_env.reset()[0]  # Adjusted for Gym 0.26+

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        result = test_env.step(action)

        # Dynamically handle the number of returned values based on Gym version
        if len(result) == 4:  # Gym < 0.26
            obs, rewards, done, info = result
            terminated = done
            truncated = False
        elif len(result) == 5:  # Gym >= 0.26
            obs, rewards, terminated, truncated, info = result
        else:
            raise ValueError("Unexpected number of return values from env.step()")

        test_env.envs[0].render()

        if terminated or truncated:
            obs = test_env.reset()[0]  # Adjusted for Gym 0.26+

    test_env.close()



