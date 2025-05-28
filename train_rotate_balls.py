# train_rotate_balls.py (第二次更正)
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import os
import time
import torch
from typing import Optional, Dict, Any, Tuple, List
import argparse
import torch.nn as nn
import sys

import mujoco
from mujoco import MjModel, MjData
from stable_baselines3 import PPO, SAC # 保留 SAC 导入，虽然主函数只用 PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy


class RotateBallsEnv(gym.Env):
    """
    灵巧手旋转两个球体的环境 (修改目标为交换位置)
    目标：控制灵巧手使两个球体交换位置
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, xml_file='rm_75_6f_description.xml', render_mode: Optional[str] = None):
        super().__init__()

        # 加载XML模型
        full_path = os.path.abspath(os.path.join(os.getcwd(), xml_file))
        print(f"加载模型：{full_path}")

        if not os.path.exists(full_path):
            raise IOError(f"文件 {full_path} 不存在")

        # 加载MuJoCo模型
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)

        # 获取球体和灵巧手的ID
        self.ball1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball1")
        self.ball2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball2")
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "R_hand_base_link")

        # 获取球体关节的ID
        self.ball1_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball1_free")
        self.ball2_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball2_free")

        # print(f"Ball1 body ID: {self.ball1_id}, joint ID: {self.ball1_joint_id}")
        # print(f"Ball2 body ID: {self.ball2_id}, joint ID: {self.ball2_joint_id}")

        # 获取手部关节的ID (根据你的XML定义) - 用于设置动作和观测
        self.hand_joints_names = [
            "R_thumb_MCP_joint1", "R_thumb_MCP_joint2",
            "R_index_MCP_joint", "R_middle_MCP_joint",
            "R_ring_MCP_joint", "R_pinky_MCP_joint"
        ]
        self.actuator_ids = []
        self.hand_joint_qpos_ids = []
        self.hand_joint_qvel_ids = []

        for name in self.hand_joints_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id != -1:
                self.actuator_ids.append(actuator_id)
                joint_id = self.model.actuator_trnid[actuator_id, 0]
                if joint_id != -1 and joint_id < self.model.njnt:
                    self.hand_joint_qpos_ids.append(self.model.jnt_qposadr[joint_id])
                    self.hand_joint_qvel_ids.append(self.model.jnt_dofadr[joint_id])
                else:
                     print(f"警告: 未找到执行器 {name} 对应的有效 joint")
            else:
                 print(f"警告: 未找到执行器 {name}")

        print(f"找到 {len(self.actuator_ids)} 个可控制的关节执行器")

        # === 添加质量和重力检查打印 ===
        # print(f"XML 中 ball1 的 body ID: {self.ball1_id}")
        # print(f"XML 中 ball2 的 body ID: {self.ball2_id}")

        # if self.ball1_id != -1:
        #     print(f"模型加载后 ball1 的质量: {self.model.body_mass[self.ball1_id]} (期望值 1.0)")
        # else:
        #     print("警告: 未找到 body 'ball1'")

        # if self.ball2_id != -1:
        #      print(f"模型加载后 ball2 的质量: {self.model.body_mass[self.ball2_id]} (期望值 1.0)")
        # else:
        #     print("警告: 未找到 body 'ball2'")

        # # 检查重力设置
        # print(f"模型加载后重力设置: {self.model.opt.gravity}")
        # # === 打印语句结束 ===

        # 动作空间: 每个关节的控制信号 (范围[-1,1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.actuator_ids),), # 动作空间维度与可控关节数一致
            dtype=np.float32
        )

        # 观测空间 (简化，只包含球体和手的位置/速度)
        # 2 * (球位置 + 球速度) + (手位置 + 手速度) + (手到球1 + 手到球2) + 重力向量 + 球体z轴速度
        obs_dim = (
            3 * 2 + 3 * 2 + # 球体位置和线速度
            3 + 3 +         # 手部位置和线速度
            3 * 2 +         # 手到两球的相对位置
            3 +             # 重力向量
            2               # 两球的z轴速度
        )

        self.observation_space = spaces.Box(
            low=-np.inf, # 可以根据实际值范围调整 low/high
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 打印观察空间维度，便于调试
        print(f"观察空间维度: {obs_dim}")

        # 渲染器
        self.render_mode = render_mode
        self.viewer = None

        # 记录交换状态
        self.last_order = None # True: ball1_y > ball2_y, False: ball1_y < ball2_y
        self.exchange_count = 0
        self.step_count = 0 # 增加步数计数器
        self._real_exchange_check = False # 交换检查标志，避免环境重置后的第一次误判

        # 初始化随机数生成器
        self.np_random = None
        self.seed() # 调用 seed 设置 np_random

        # 动作噪音 (减小)
        self.action_noise = 0.03

        # 初始化状态跟踪变量
        self.sim_errors = 0  # 记录物理模拟错误
        self.training_started = time.time()  # 记录训练开始时间

        # 机械臂固定位置 (如果需要，可在reset中设置 root1 的 qpos)
        # self.arm_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root1") # 已在前面获取ID
        self.fixed_arm_pos = np.array([-0.28, 0.0, 0.1])
        self.fixed_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])


    def seed(self, seed=None):
        """设置随机种子"""
        # gym.Env 的 seed 方法现在返回一个 list
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_obs(self) -> np.ndarray:
        """获取观察状态"""
        try:
            # # 1. 手部关节角度和速度 (如果包含在观测空间中)
            # qpos_hand = np.zeros(len(self.actuator_ids), dtype=np.float32)
            # qvel_hand = np.zeros(len(self.actuator_ids), dtype=np.float32)

            # # 获取关节角度和速度（使用存储的 qpos/qvel ID）
            # for i, qpos_id in enumerate(self.hand_joint_qpos_ids):
            #      if qpos_id != -1 and qpos_id < len(self.data.qpos):
            #          qpos_hand[i] = self.data.qpos[qpos_id]

            # for i, qvel_id in enumerate(self.hand_joint_qvel_ids):
            #      if qvel_id != -1 and qvel_id < len(self.data.qvel):
            #          qvel_hand[i] = self.data.qvel[qvel_id]


            # 2. 球体位置和速度
            ball1_pos = np.array(self.data.xpos[self.ball1_id], dtype=np.float32) if self.ball1_id != -1 and self.ball1_id < len(self.data.xpos) else np.zeros(3)
            ball2_pos = np.array(self.data.xpos[self.ball2_id], dtype=np.float32) if self.ball2_id != -1 and self.ball2_id < len(self.data.xpos) else np.zeros(3)

            # 安全地获取线速度 (cvel 的前3个元素)
            ball1_vel = np.array(self.data.cvel[self.ball1_id][:3], dtype=np.float32) if self.ball1_id != -1 and self.ball1_id < len(self.data.cvel) and len(self.data.cvel[self.ball1_id]) >= 3 else np.zeros(3)
            ball2_vel = np.array(self.data.cvel[self.ball2_id][:3], dtype=np.float32) if self.ball2_id != -1 and self.ball2_id < len(self.data.cvel) and len(self.data.cvel[self.ball2_id]) >= 3 else np.zeros(3)

            # 3. 手部位置和线速度
            hand_pos = np.array(self.data.xpos[self.hand_id], dtype=np.float32) if self.hand_id != -1 and self.hand_id < len(self.data.xpos) else np.zeros(3)
            hand_vel = np.array(self.data.cvel[self.hand_id][:3], dtype=np.float32) if self.hand_id != -1 and self.hand_id < len(self.data.cvel) and len(self.data.cvel[self.hand_id]) >= 3 else np.zeros(3)

            # 计算手到球的相对位置
            hand_to_ball1 = ball1_pos - hand_pos
            hand_to_ball2 = ball2_pos - hand_pos
            
            # 添加重力向量信息
            gravity = np.array(self.model.opt.gravity, dtype=np.float32) if hasattr(self.model, 'opt') else np.array([0, 0, -9.81], dtype=np.float32)
            
            # 提取球体z轴速度（突出显示重力影响）
            ball1_z_vel = ball1_vel[2]
            ball2_z_vel = ball2_vel[2]

            # 合并观察
            obs = np.concatenate([
                # qpos_hand, # 关节角度 (如果包含在观测空间)
                # qvel_hand, # 关节速度 (如果包含在观测空间)
                ball1_pos,      # 球1位置
                ball1_vel,      # 球1速度
                ball2_pos,      # 球2位置
                ball2_vel,      # 球2速度
                hand_pos,       # 手部位置
                hand_vel,       # 手部速度
                hand_to_ball1,  # 手到球1的向量
                hand_to_ball2,  # 手到球2的向量
                gravity,        # 重力向量
                np.array([ball1_z_vel, ball2_z_vel])  # 球体z轴速度，突出重力影响
            ]).astype(np.float32)

            return obs

        except Exception as e:
            print(f"获取观察时出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回全零数组作为默认观察
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _set_action(self, action: np.ndarray) -> None:
        """执行动作，控制灵巧手"""
        try:
            # 将动作从[-1,1]映射到实际的控制范围
            # 假设 actuator_ids 列表中的顺序对应 action 数组的顺序
            if len(action) != len(self.actuator_ids):
                return

            ctrl = np.zeros(self.model.nu, dtype=np.float32)

            for i, actuator_id in enumerate(self.actuator_ids):
                 if actuator_id != -1:
                    # 获取控制范围
                    ctrl_range_min = self.model.actuator_ctrlrange[actuator_id, 0]
                    ctrl_range_max = self.model.actuator_ctrlrange[actuator_id, 1]

                    # 添加少量噪声增加探索
                    noisy_action = action[i] + self.np_random.normal(0, 0.03)  # 减小噪声大小
                    noisy_action = np.clip(noisy_action, -1.0, 1.0)

                    # 映射到实际范围
                    ctrl[actuator_id] = 0.5 * (noisy_action + 1.0) * (ctrl_range_max - ctrl_range_min) + ctrl_range_min
            # 设置控制信号
            self.data.ctrl[:] = ctrl

        except Exception as e:
            print(f"设置动作时出错: {e}")
            import traceback
            traceback.print_exc()

    def _compute_reward(self) -> float:
        """
        强化交换行为的奖励函数
        """
        try:
            if self.ball1_id == -1 or self.ball2_id == -1 or \
               self.ball1_id >= len(self.data.xpos) or self.ball2_id >= len(self.data.xpos):
                return 0.0

            ball1_pos = self.data.xpos[self.ball1_id]
            ball2_pos = self.data.xpos[self.ball2_id]
            ball1_y = ball1_pos[1]
            ball2_y = ball2_pos[1]
            
            # 确保每次计算奖励时都重置为0
            reward = 0.0
            
            # 球体相对位置顺序
            current_order = ball1_y > ball2_y
            
            # 获取手部位置，用于后续计算
            hand_pos = self.data.xpos[self.hand_id]
            
            # === 新增：球体在手掌上的稳定性奖励 ===
            # 定义手掌区域
            palm_height = hand_pos[2]  # 手掌高度
            palm_radius = 0.12  # 手掌有效半径
            min_height = palm_height + 0.01  # 球体应该在手掌上方一点
            ideal_height = palm_height + 0.06  # 理想的球体高度
            
            # 计算球体是否在手掌上方
            ball1_xy_dist = np.linalg.norm(ball1_pos[:2] - hand_pos[:2])
            ball2_xy_dist = np.linalg.norm(ball2_pos[:2] - hand_pos[:2])
            ball1_on_palm = ball1_xy_dist < palm_radius and ball1_pos[2] > min_height
            ball2_on_palm = ball2_xy_dist < palm_radius and ball2_pos[2] > min_height
            
            # 球体在手掌上的奖励
            if ball1_on_palm:
                palm_reward1 = 50.0 * np.exp(-5.0 * abs(ball1_pos[2] - ideal_height))
                reward += palm_reward1
            else:
                # 不在手掌上的惩罚，距离越远惩罚越大
                palm_penalty1 = -80.0 * (1.0 - np.exp(-3.0 * max(0, ball1_xy_dist - palm_radius)))
                reward += palm_penalty1
                
            if ball2_on_palm:
                palm_reward2 = 50.0 * np.exp(-5.0 * abs(ball2_pos[2] - ideal_height))
                reward += palm_reward2
            else:
                # 不在手掌上的惩罚，距离越远惩罚越大
                palm_penalty2 = -80.0 * (1.0 - np.exp(-3.0 * max(0, ball2_xy_dist - palm_radius)))
                reward += palm_penalty2
            
            # 球体高度过低的严厉惩罚，防止掉落
            fall_threshold = palm_height - 0.05  # 低于手掌一定高度视为要掉落
            if ball1_pos[2] < fall_threshold:
                fall_penalty = -500.0 * (fall_threshold - ball1_pos[2]) * 20.0  # 极大增加惩罚力度
                reward += fall_penalty
                # print(f"球1高度过低警告: {ball1_pos[2]:.3f}, 惩罚: {fall_penalty:.1f}")
                
            if ball2_pos[2] < fall_threshold:
                fall_penalty = -500.0 * (fall_threshold - ball2_pos[2]) * 20.0  # 极大增加惩罚力度
                reward += fall_penalty
                # print(f"球2高度过低警告: {ball2_pos[2]:.3f}, 惩罚: {fall_penalty:.1f}")
                
            # 球体离开手掌区域的严厉惩罚
            if not ball1_on_palm:
                off_palm_penalty = -200.0 * (1.0 - np.exp(-5.0 * max(0, ball1_xy_dist - palm_radius)))
                reward += off_palm_penalty
                # if ball1_xy_dist > palm_radius * 1.2:  # 明显离开手掌
                #     print(f"球1离开手掌区域: 距离={ball1_xy_dist:.3f}, 惩罚={off_palm_penalty:.1f}")
                
            if not ball2_on_palm:
                off_palm_penalty = -200.0 * (1.0 - np.exp(-5.0 * max(0, ball2_xy_dist - palm_radius)))
                reward += off_palm_penalty
                # if ball2_xy_dist > palm_radius * 1.2:  # 明显离开手掌
                #     print(f"球2离开手掌区域: 距离={ball2_xy_dist:.3f}, 惩罚={off_palm_penalty:.1f}")
            
            # === 原有奖励部分 ===
            # 1. 交换奖励 - 增加交换阈值检测
            y_diff = abs(ball1_y - ball2_y)
            if self.last_order is not None and current_order != self.last_order and y_diff > 0.01:  # 添加最小距离要求
                # 检查是否是真实的交换（非环境重置后的第一次检测）
                if hasattr(self, '_real_exchange_check') and self._real_exchange_check:
                    # 只有当两球都在手掌上方时才给予交换奖励
                    if ball1_on_palm and ball2_on_palm:
                        exchange_reward = 2000.0  # 进一步提高交换奖励
                        self.exchange_count += 1
                        reward += exchange_reward
                        if self.exchange_count > 1:
                            combo_reward = min(500.0 * (self.exchange_count - 1), 2000.0)  # 增加连续交换奖励
                            reward += combo_reward
                        print(f"成功交换球体位置! 交换次数: {self.exchange_count}, 奖励: {exchange_reward}")
                    else:
                        # 如果球不在手掌上，交换不算有效
                        print("球体位置发生变化，但不在手掌上，不计为有效交换")
                else:
                    # 第一次检查不计算奖励，只更新状态
                    print(f"首次位置检查，记录排序状态: ball1_y({ball1_y:.4f}) {'>' if current_order else '<'} ball2_y({ball2_y:.4f})")
                    self._real_exchange_check = True
            
            # 无论如何都更新last_order状态
            self.last_order = current_order

            # 2. 手部精确控制奖励 - 鼓励手指协调运动
            # 计算手部到两球的距离
            dist1 = np.linalg.norm(ball1_pos - hand_pos)
            dist2 = np.linalg.norm(ball2_pos - hand_pos)
            
            # 鼓励手部同时接近两球（盘核桃的关键）
            optimal_dist = 0.08  # 理想的控制距离
            control_reward1 = 15.0 * np.exp(-10.0 * abs(dist1 - optimal_dist))
            control_reward2 = 15.0 * np.exp(-10.0 * abs(dist2 - optimal_dist))
            reward += control_reward1 + control_reward2
            
            # 额外奖励：当两球都在理想距离内时
            if dist1 < optimal_dist + 0.02 and dist2 < optimal_dist + 0.02:
                reward += 30.0

            # 3. 改进的交叉运动检测
            crossing_direction = False
            if hasattr(self, '_last_ball1_pos') and hasattr(self, '_last_ball2_pos'):
                last_ball1_y = self._last_ball1_pos[1]
                last_ball2_y = self._last_ball2_pos[1]
                
                # 计算运动方向和速度
                ball1_dy = ball1_y - last_ball1_y
                ball2_dy = ball2_y - last_ball2_y
                
                # 更精确的交叉检测：球体朝相反方向移动且有足够速度
                if abs(ball1_dy) > 0.0005 and abs(ball2_dy) > 0.0005:  # 确保有明显移动
                    if (ball1_y > ball2_y and ball1_dy < 0 and ball2_dy > 0) or \
                       (ball1_y < ball2_y and ball1_dy > 0 and ball2_dy < 0):
                        crossing_direction = True
            
            # 交叉运动奖励 - 只有当球在手掌上时才给予交叉奖励
            if crossing_direction and ball1_on_palm and ball2_on_palm:
                cross_reward = 100.0 * np.exp(-15.0 * y_diff)  # 距离越近奖励越高
                reward += cross_reward
                
                # 临界交叉奖励
                if y_diff < 0.02:
                    reward += 150.0
                    print(f"球体临界交叉! y_diff={y_diff:.4f}")

            # 4. 手部运动协调性奖励
            if len(self.actuator_ids) > 0:
                # 获取手部关节速度
                joint_velocities = []
                for qvel_id in self.hand_joint_qvel_ids:
                    if qvel_id != -1 and qvel_id < len(self.data.qvel):
                        joint_velocities.append(abs(self.data.qvel[qvel_id]))
                
                if joint_velocities:
                    # 鼓励适度的关节运动（不要太快也不要太慢）
                    avg_joint_vel = np.mean(joint_velocities)
                    optimal_joint_vel = 0.5  # 理想关节速度
                    coordination_reward = 10.0 * np.exp(-5.0 * abs(avg_joint_vel - optimal_joint_vel))
                    reward += coordination_reward

            # 5. 球体相对运动奖励 - 专注于交换意图
            if hasattr(self, '_last_ball1_pos') and hasattr(self, '_last_ball2_pos'):
                # 计算球体间的相对位置变化
                last_rel_pos = self._last_ball1_pos[1] - self._last_ball2_pos[1]
                curr_rel_pos = ball1_pos[1] - ball2_pos[1]
                
                # 如果相对位置在变小（朝着交叉方向）
                if abs(curr_rel_pos) < abs(last_rel_pos):
                    # 相对位置变化的绝对值
                    rel_pos_change = abs(abs(curr_rel_pos) - abs(last_rel_pos))
                    rel_motion_reward = 50.0 * rel_pos_change  # 增加朝交叉方向移动的奖励
                    reward += rel_motion_reward
                    
                    # 当相对位置接近零或改变符号时，给予额外奖励
                    if curr_rel_pos * last_rel_pos <= 0:  # 符号变化或为零
                        reward += 80.0
                        print("球体相对位置发生符号变化，接近交换!")
                else:
                    # 对远离交叉的移动增加惩罚
                    rel_pos_change = abs(abs(curr_rel_pos) - abs(last_rel_pos))
                    reward -= 20.0 * rel_pos_change  # 增加惩罚力度

            # 6. 球体高度维持奖励 - 确保球体不会掉落但不要太高
            max_height = 0.9  # 最大允许高度
            height_penalty = 0.0
            
            # 给予高度太高的惩罚 (增加惩罚力度)
            if ball1_pos[2] > max_height:
                height_penalty += 30.0 * (ball1_pos[2] - max_height) * 10.0
                print(f"球1高度过高: {ball1_pos[2]:.3f}, 惩罚: {30.0 * (ball1_pos[2] - max_height) * 10.0:.1f}")
            
            if ball2_pos[2] > max_height:
                height_penalty += 30.0 * (ball2_pos[2] - max_height) * 10.0
                print(f"球2高度过高: {ball2_pos[2]:.3f}, 惩罚: {30.0 * (ball2_pos[2] - max_height) * 10.0:.1f}")
            
            # 更强的高度维持奖励 - 使用前面定义的理想高度
            height_reward = 20.0 * (np.exp(-8.0 * abs(ball1_pos[2] - ideal_height)) + 
                                   np.exp(-8.0 * abs(ball2_pos[2] - ideal_height)))
            reward += height_reward - height_penalty
            
            # 7. 两球高度相似奖励 - 鼓励两球在同一平面上交换
            height_diff = abs(ball1_pos[2] - ball2_pos[2])
            same_height_reward = 25.0 * np.exp(-10.0 * height_diff)
            reward += same_height_reward
            
            # 8. 适应重力的垂直运动奖励
            # 获取球体的Z轴速度
            ball1_z_vel = 0.0
            ball2_z_vel = 0.0
            if self.ball1_id != -1 and self.ball1_id < len(self.data.cvel) and len(self.data.cvel[self.ball1_id]) >= 3:
                ball1_z_vel = self.data.cvel[self.ball1_id][2]
            if self.ball2_id != -1 and self.ball2_id < len(self.data.cvel) and len(self.data.cvel[self.ball2_id]) >= 3:
                ball2_z_vel = self.data.cvel[self.ball2_id][2]
            
            # 如果在理想高度附近，鼓励小的垂直速度（稳定）
            if abs(ball1_pos[2] - ideal_height) < 0.1:
                z_vel_reward1 = 10.0 * np.exp(-5.0 * abs(ball1_z_vel))
                reward += z_vel_reward1
            
            if abs(ball2_pos[2] - ideal_height) < 0.1:
                z_vel_reward2 = 10.0 * np.exp(-5.0 * abs(ball2_z_vel))
                reward += z_vel_reward2
            
            # 如果高度太高，鼓励向下运动
            if ball1_pos[2] > ideal_height + 0.1 and ball1_z_vel < 0:
                reward += 5.0 * abs(ball1_z_vel)  # 向下运动得到奖励
            
            if ball2_pos[2] > ideal_height + 0.1 and ball2_z_vel < 0:
                reward += 5.0 * abs(ball2_z_vel)  # 向下运动得到奖励
            
            # 如果高度太低，鼓励向上运动
            if ball1_pos[2] < ideal_height - 0.1 and ball1_z_vel > 0:
                reward += 5.0 * ball1_z_vel  # 向上运动得到奖励
            
            if ball2_pos[2] < ideal_height - 0.1 and ball2_z_vel > 0:
                reward += 5.0 * ball2_z_vel  # 向上运动得到奖励

            # 更新球体位置记录
            self._last_ball1_pos = ball1_pos.copy()
            self._last_ball2_pos = ball2_pos.copy()
            
            # 奖励裁剪
            reward = np.clip(reward, -3000.0, 3000.0)  # 扩大惩罚范围到最大值
            return float(reward)
        except Exception as e:
            print(f"计算奖励时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作
        参数: action - 动作向量
        返回: (obs, reward, terminated, truncated, info)
        """
        try:
            self.step_count += 1  # 增加步数计数
            truncated = False
            # 确保奖励值初始化为0
            reward = 0.0

            # 限制动作范围
            action = np.clip(action, -1, 1)

            # 设置控制信号
            self._set_action(action)

            # 模拟一步
            try:
                mujoco.mj_step(self.model, self.data)
            except Exception as e:
                print(f"mj_step 出错: {e}")
                # 如果模拟出错，返回终止信号
                obs = self._get_obs()
                return obs, -1.0, True, False, {'termination_reason': '物理模拟错误'}

            # 再次执行前向动力学以更新位置 (确保观察和奖励基于最新状态)
            try:
                mujoco.mj_forward(self.model, self.data)
            except Exception as e:
                print(f"mj_forward 出错: {e}")
                # 如果前向动力学出错，返回终止信号
                obs = self._get_obs()
                return obs, -1.0, True, False, {'termination_reason': '物理模拟错误'}

            # 获取观察
            obs = self._get_obs()

            # 先检查球体是否掉落到地面 - 如果掉落则立即终止
            # 定义地面高度阈值 - 当球体高度低于此值时视为掉落
            ground_height = 0.1  # 提高地面高度阈值，更早检测到掉落

            # 获取球体高度和位置 (安全访问 xpos)
            ball1_pos = self.data.xpos[self.ball1_id].copy() if self.ball1_id != -1 and self.ball1_id < len(self.data.xpos) else np.zeros(3)
            ball2_pos = self.data.xpos[self.ball2_id].copy() if self.ball2_id != -1 and self.ball2_id < len(self.data.xpos) else np.zeros(3)
            ball1_height = ball1_pos[2]
            ball2_height = ball2_pos[2]

            # 记录球体上一步的位置
            if not hasattr(self, '_last_ball1_pos'):
                self._last_ball1_pos = np.copy(ball1_pos)
            if not hasattr(self, '_last_ball2_pos'):
                self._last_ball2_pos = np.copy(ball2_pos)
            if not hasattr(self, '_no_move_steps'):
                self._no_move_steps = 0

            # 检查球体是否掉落到地面
            terminated = False
            termination_reason = None
            termination_penalty = 0.0

            if (self.ball1_id != -1 and ball1_height < ground_height) or \
               (self.ball2_id != -1 and ball2_height < ground_height):
                 terminated = True
                 termination_reason = "有球掉落地面"
                 termination_penalty = -3000.0  # 最大惩罚
                 print(f"球体掉落地面! 球1高度: {ball1_height:.3f}, 球2高度: {ball2_height:.3f}")
                 
            # 获取手部位置，检查球体是否从手掌上掉落
            hand_pos = self.data.xpos[self.hand_id] if self.hand_id != -1 and self.hand_id < len(self.data.xpos) else np.zeros(3)
            palm_height = hand_pos[2]  # 手掌高度
            fall_threshold = palm_height - 0.1  # 低于手掌一定高度视为掉落
            
            # 检查球体是否从手掌上掉落
            if not terminated and ((self.ball1_id != -1 and ball1_height < fall_threshold) or \
                                  (self.ball2_id != -1 and ball2_height < fall_threshold)):
                terminated = True
                termination_reason = "球体从手掌上掉落"
                termination_penalty = -2000.0  # 最大惩罚
                print(f"球体从手掌上掉落! 球1高度: {ball1_height:.3f}, 球2高度: {ball2_height:.3f}, 手掌高度: {palm_height:.3f}")
                
            # 检查球体是否离手掌太远（水平方向）
            palm_radius = 0.15  # 手掌有效半径
            ball1_xy_dist = np.linalg.norm(ball1_pos[:2] - hand_pos[:2]) if 'ball1_pos' in locals() else 0
            ball2_xy_dist = np.linalg.norm(ball2_pos[:2] - hand_pos[:2]) if 'ball2_pos' in locals() else 0
            
            # 如果球体水平方向离开手掌太远，也视为掉落
            if not terminated and (ball1_xy_dist > palm_radius * 2.0 or ball2_xy_dist > palm_radius * 2.0):
                terminated = True
                termination_reason = "球体水平方向离开手掌太远"
                termination_penalty = -1500.0  # 最大惩罚
                print(f"球体水平方向离开手掌太远! 球1距离: {ball1_xy_dist:.3f}, 球2距离: {ball2_xy_dist:.3f}, 手掌半径: {palm_radius:.3f}")

            # 检查球体是否长时间未移动
            move_threshold = 0.0001  # 降低移动阈值，更严格要求球体移动
            no_move_limit = 500  # 增加无移动步数限制，更宽松
            ball1_move = np.linalg.norm(ball1_pos - self._last_ball1_pos)
            ball2_move = np.linalg.norm(ball2_pos - self._last_ball2_pos)
            
            # 同时考虑位置和速度判断是否真正静止
            ball1_vel_norm = np.linalg.norm(self.data.cvel[self.ball1_id][:3]) if self.ball1_id != -1 and self.ball1_id < len(self.data.cvel) else 0
            ball2_vel_norm = np.linalg.norm(self.data.cvel[self.ball2_id][:3]) if self.ball2_id != -1 and self.ball2_id < len(self.data.cvel) else 0
            
            # === 静止检测相关代码已被注释/删除 ===
            # really_not_moving = (ball1_move < move_threshold and ball2_move < move_threshold and 
            #                     ball1_vel_norm < 0.001 and ball2_vel_norm < 0.001)
            # moved = (ball1_move >= move_threshold or ball2_move >= move_threshold)
            # now = time.time()
            # if not hasattr(self, '_no_move_time_start'):
            #     self._no_move_time_start = now
            # if moved:
            #     self._no_move_steps = 0
            #     self._no_move_time_start = now
            # elif really_not_moving:  # 只有真正静止时才增加计数
            #     self._no_move_steps += 1
            # # 检查无移动终止条件
            # # 1. 时间超过5秒真正未移动
            # if really_not_moving and (now - self._no_move_time_start) > 5.0:
            #     terminated = True
            #     termination_reason = "超过5秒真正未移动"
            #     termination_penalty = -10.0
            #     print(f"球体超过5秒真正未移动，回合终止")
            # # 2. 步数超过no_move_limit步真正未移动
            # elif really_not_moving and self._no_move_steps > no_move_limit:
            #     terminated = True
            #     termination_reason = f"超过{no_move_limit}步真正未移动"
            #     termination_penalty = -10.0
            #     print(f"球体超过{no_move_limit}步真正未移动，回合终止")
            # # 3. 超时终止 - 如果长时间很少移动但还是有微小移动，也终止
            # elif self.step_count > 500 and (now - self._no_move_time_start) > 5.0 and \
            #      ball1_move < 0.001 and ball2_move < 0.001:  # 移动很小但不是完全静止
            #     terminated = True
            #     termination_reason = "超过5秒移动很小，防止卡住"
            #     termination_penalty = -5.0  # 较小的惩罚
            #     print(f"球体移动很小且超过5秒，防止卡住，回合终止")
            # === 静止检测相关代码结束 ===
            # 更新上一步位置
            self._last_ball1_pos = np.copy(ball1_pos)
            self._last_ball2_pos = np.copy(ball2_pos)

            # 球体飞得太高惩罚
            max_height = 1.5  # 降低最大高度限制
            if (self.ball1_id != -1 and ball1_height > max_height) or \
               (self.ball2_id != -1 and ball2_height > max_height):
                 termination_reason = "球体飞得太高，超出控制范围"
                 termination_penalty = -20.0  # 增加高度惩罚
                 terminated = True  # 改为直接终止
                 print(f"球体飞得太高! 球1高度: {ball1_height:.3f}, 球2高度: {ball2_height:.3f}，回合终止")

            # 球体离手太远惩罚 
            max_xy_dist = 0.5  # 减小最大允许距离
            hand_pos = self.data.xpos[self.hand_id] if self.hand_id != -1 and self.hand_id < len(self.data.xpos) else np.zeros(3)
            ball1_xy_dist = np.linalg.norm(ball1_pos[:2] - hand_pos[:2]) if 'ball1_pos' in locals() else 0
            ball2_xy_dist = np.linalg.norm(ball2_pos[:2] - hand_pos[:2]) if 'ball2_pos' in locals() else 0
            if ball1_xy_dist > max_xy_dist or ball2_xy_dist > max_xy_dist:
                termination_reason = "球体离手太远"
                termination_penalty = -20.0
                terminated = True  # 改为直接终止
                print(f"球体离手太远! 球1距离: {ball1_xy_dist:.3f}, 球2距离: {ball2_xy_dist:.3f}，回合终止")

            # 超时终止 - 防止回合过长
            max_steps = 8000  # 减少最大步数限制
            if self.step_count >= max_steps:
                terminated = True
                termination_reason = "回合步数达到上限"
                # 根据交换次数给予不同的奖励/惩罚
                if self.exchange_count > 0:
                    termination_penalty = 0.0  # 如果有交换，不惩罚
                else:
                    termination_penalty = -20.0  # 如果一次也没交换，增加惩罚
                print(f"回合达到最大步数 {max_steps}，交换次数: {self.exchange_count}")
            
            # 成功完成任务提前终止 - 当交换次数达到目标时提前结束
            elif self.exchange_count >= 3:  # 成功交换3次
                terminated = True
                termination_reason = "成功完成交换任务"
                termination_penalty = 200.0  # 额外奖励
                print(f"成功完成交换任务！共交换{self.exchange_count}次，提前结束回合")

            # 计算奖励 (如果已经终止，可以跳过)
            reward = self._compute_reward() if not terminated else 0.0

            # 添加惩罚到奖励中（只要有惩罚就加）
            if termination_penalty != 0.0:
                # 如果有终止惩罚，确保奖励重置为惩罚值
                if terminated:
                    reward = termination_penalty
                else:
                    reward += termination_penalty

            # 信息字典
            info = {
                'termination_reason': termination_reason,
                'exchange_count': self.exchange_count,
                'ball1_pos': self.data.xpos[self.ball1_id].copy() if self.ball1_id != -1 and self.ball1_id < len(self.data.xpos) else np.zeros(3),
                'ball2_pos': self.data.xpos[self.ball2_id].copy() if self.ball2_id != -1 and self.ball2_id < len(self.data.xpos) else np.zeros(3),
                'ball1_vel': self.data.cvel[self.ball1_id][:3].copy() if self.ball1_id != -1 and self.ball1_id < len(self.data.cvel) and len(self.data.cvel[self.ball1_id]) >= 3 else np.zeros(3),
                'ball2_vel': self.data.cvel[self.ball2_id][:3].copy() if self.ball2_id != -1 and self.ball2_id < len(self.data.cvel) and len(self.data.cvel[self.ball2_id]) >= 3 else np.zeros(3),
                'steps': self.step_count,
            }

            # 如果回合结束，添加stable_baselines3需要的特殊键值用于记录
            if terminated or truncated:
                info['episode'] = {
                    'r': reward,  # 最终奖励
                    'l': self.step_count,  # 回合长度
                    'exchange_count': self.exchange_count,
                }
                # 回合结束时重置步数计数器
                self.step_count = 0

            # gym recommends returning terminated and truncated separately
            return obs, reward, terminated, truncated, info

        except Exception as e:
            print(f"执行步骤时出错: {e}")
            import traceback
            traceback.print_exc()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            # 出错时标记为 terminated 结束回合
            return obs, 0.0, True, False, {'episode': {'r': 0.0, 'l': 0, 'exchange_count': self.exchange_count}}


    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境：确保球体位置正确初始化，并提供更多随机性
        返回: (obs, info)
        """
        try:
            # 重置交换状态和记录变量
            self.exchange_count = 0
            self.step_count = 0
            self.last_order = None  # 重置上一次的排序状态
            self._real_exchange_check = False  # 重置交换检查标志
            
            if seed is not None:
                self.seed(seed) # 调用自定义的 seed 方法

            # 强制重力修正 - 增大重力
            if hasattr(self.model, 'opt'):
                self.model.opt.gravity[:] = [0, 0, -9.81 * 1.5]  # 增大重力为原来的1.5倍
                print(f"设置重力为: {self.model.opt.gravity}")
                mujoco.mj_forward(self.model, self.data)
            
            # 1. 使用mujoco原生重置
            mujoco.mj_resetData(self.model, self.data)
            
            # 可选：如果有关键帧，可以重置到指定关键帧
            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            # 2. 设置球体的位置和速度 - 让球体刚好落在手上表面
            hand_z = 0.64
            ball_radius = 0.017
            gap = 0.001
            # 选择交错的起始位置 - 随机选择初始排序
            if self.np_random.random() < 0.5:
                base1 = np.array([-0.28, 0.024, hand_z + ball_radius + gap])
                base2 = np.array([-0.28, -0.024, hand_z + ball_radius + gap])
            else:
                base1 = np.array([-0.28, -0.024, hand_z + ball_radius + gap])
                base2 = np.array([-0.28, 0.024, hand_z + ball_radius + gap])
            pos_noise = 0.002
            ball1_init_pos = base1 + np.concatenate([self.np_random.uniform(-pos_noise, pos_noise, 2), [0.0]])
            ball2_init_pos = base2 + np.concatenate([self.np_random.uniform(-pos_noise, pos_noise, 2), [0.0]])

            # 3. 设置球体1位置
            if self.ball1_joint_id != -1:
                qpos_start = self.model.jnt_qposadr[self.ball1_joint_id]
                if qpos_start >= 0 and qpos_start + 7 <= len(self.data.qpos):
                    self.data.qpos[qpos_start:qpos_start+3] = ball1_init_pos
                    self.data.qpos[qpos_start+3:qpos_start+7] = [1.0, 0.0, 0.0, 0.0]
                    qvel_start = self.model.jnt_dofadr[self.ball1_joint_id]
                    if qvel_start >= 0 and qvel_start + 6 <= len(self.data.qvel):
                        self.data.qvel[qvel_start:qvel_start+6] = 0.0
            # 4. 设置球体2位置
            if self.ball2_joint_id != -1:
                qpos_start = self.model.jnt_qposadr[self.ball2_joint_id]
                if qpos_start >= 0 and qpos_start + 7 <= len(self.data.qpos):
                    self.data.qpos[qpos_start:qpos_start+3] = ball2_init_pos
                    self.data.qpos[qpos_start+3:qpos_start+7] = [1.0, 0.0, 0.0, 0.0]
                    qvel_start = self.model.jnt_dofadr[self.ball2_joint_id]
                    if qvel_start >= 0 and qvel_start + 6 <= len(self.data.qvel):
                        self.data.qvel[qvel_start:qvel_start+6] = 0.0
            print("球体初始位置已设置(落在手上)")

            # # 5. 设置手部关节初始位置 - 使手指处于半握状态，便于操控球体
            # try:
            #     for name, data in self.hand_joints.items():
            #         joint_id = data["joint_id"]
            #         if joint_id < self.data.qpos.shape[0]:
            #             low, high = self.model.jnt_range[joint_id]
            #             if "thumb" in name:
            #                 pos = low + 0.7 * (high - low)
            #             elif "index" in name:
            #                 pos = low + 0.5 * (high - low)
            #             else:
            #                 pos = low + 0.6 * (high - low)
            #             pos += self.np_random.uniform(-0.03, 0.03) * (high - low)
            #             self.data.qpos[joint_id] = np.clip(pos, low, high)
            #     print("手部关节初始位置已设置")
            # except Exception as e:
            #     print(f"设置手部关节初始位置时出错: {e}")
            mujoco.mj_forward(self.model, self.data)
            
            # 初始化存储球体位置
            if self.ball1_id < len(self.data.xpos) and self.ball2_id < len(self.data.xpos):
                self.prev_ball1_pos = self.data.xpos[self.ball1_id].copy()
                self.prev_ball2_pos = self.data.xpos[self.ball2_id].copy()
                self._last_ball1_pos = self.data.xpos[self.ball1_id].copy() 
                self._last_ball2_pos = self.data.xpos[self.ball2_id].copy()
                
                # 设置初始的last_order状态，避免在第一步检测到错误的交换
                ball1_y = self.data.xpos[self.ball1_id][1]
                ball2_y = self.data.xpos[self.ball2_id][1]
                self.last_order = ball1_y > ball2_y
                print(f"初始球体排序: ball1_y({ball1_y:.4f}) {'>' if self.last_order else '<'} ball2_y({ball2_y:.4f})")
            else:
                print(f"警告: 球体索引无效，无法记录初始位置")
                self.prev_ball1_pos = np.zeros(3)
                self.prev_ball2_pos = np.zeros(3)
                self._last_ball1_pos = np.zeros(3)
                self._last_ball2_pos = np.zeros(3)
            
            # 重置其他跟踪变量
            if not hasattr(self, '_no_move_steps'):
                self._no_move_steps = 0
            else:
                self._no_move_steps = 0
                
            if not hasattr(self, '_no_move_time_start'):
                self._no_move_time_start = time.time()
            else:
                self._no_move_time_start = time.time()
                
            return self._get_obs(), {}
        except Exception as e:
            print(f"重置环境时出错: {e}")
            import traceback
            traceback.print_exc()
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {}


    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            try:
                from mujoco import viewer

                if self.viewer is None:
                    self.viewer = viewer.launch_passive(self.model, self.data)
                    # 设置相机视角 - 优化视角使盘核桃动作更清晰
                    self.viewer.cam.distance = 1.0  # 稍微拉近距离
                    self.viewer.cam.elevation = -25  # 调整仰角
                    self.viewer.cam.azimuth = 90  # 侧面视角，便于观察交换动作
                    self.viewer.cam.lookat[0] = -0.3  # 聚焦点X坐标
                    self.viewer.cam.lookat[1] = 0.0  # 聚焦点Y坐标
                    self.viewer.cam.lookat[2] = 0.6  # 聚焦点Z坐标
                    
                    # 渲染设置，增强可视化效果
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0  # 关闭接触点显示
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0  # 关闭接触力显示
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0  # 关闭凸包显示
                    
                    # 增强物体轮廓，便于观察球体位置
                    self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
                    
                    # 球体轨迹
                    if hasattr(mujoco.mjtVisFlag, 'mjVIS_PARTICLE'):
                        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PARTICLE] = 1  # 显示粒子/轨迹

                if self.viewer is not None:
                    self.viewer.sync()
                    # 可以根据需要调整帧率或添加动态延迟

            except Exception as e:
                print(f"渲染时出错: {e}")
                import traceback
                traceback.print_exc()
                # 如果渲染出错，可以考虑关闭 viewer 防止阻塞
                self.close()


    def close(self):
        """关闭环境和渲染器"""
        try:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        except Exception as e:
            print(f"关闭环境时出错: {e}")
            import traceback
            traceback.print_exc()


# ==================================================
# 训练相关代码 (保持你的主要结构，只做微调)
# ==================================================

# TrainingProgressCallback 类保持原样，只修改打印内容

class TrainingProgressCallback(CheckpointCallback):
    """跟踪并打印训练进度的回调函数"""

    def __init__(self, save_freq, save_path, name_prefix="model", log_interval=1000,
                 num_envs=1, verbose=0, save_replay_buffer=False, save_vecnormalize=False):
        super().__init__(save_freq, save_path, name_prefix, verbose, save_replay_buffer, save_vecnormalize)
        self.log_interval = max(1, log_interval)
        self.num_envs = max(1, num_envs)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_exchange_counts = [] # 记录交换次数
        self.total_episodes = 0
        self.start_time = time.time()
        self._last_save_time_steps = 0 # 用于按时间步保存检查点

    def _on_step(self) -> bool:
        try:
            # 修复超时检测机制 - 增加更新计时器
            current_time = time.time()
            
            # 检查是否有处理过的回合
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                # 重置超时计数器，因为我们有进度
                if not hasattr(self, '_last_progress_time'):
                    self._last_progress_time = current_time
                else:
                    self._last_progress_time = current_time
                
                # 处理新回合信息
                num_completed_episodes = len(self.model.ep_info_buffer)
                new_episodes_count = num_completed_episodes - self.total_episodes

                if new_episodes_count > 0:
                    # 获取最新的回合信息
                    new_infos = list(self.model.ep_info_buffer)[-new_episodes_count:]

                    for info in new_infos:
                        if 'r' in info and 'l' in info: # Reward and length
                            self.episode_rewards.append(float(info['r']))
                            self.episode_lengths.append(int(info['l']))

                            # 记录交换次数
                            if 'exchange_count' in info:
                                self.episode_exchange_counts.append(int(info['exchange_count']))

                    self.total_episodes = num_completed_episodes
            else:
                # 检查是否超时 - 如果超过5分钟没有进度，可能卡住了
                timeout_limit = 300  # 5分钟超时
                if hasattr(self, '_last_progress_time') and current_time - self._last_progress_time > timeout_limit:
                    print(f"\n警告: 训练可能卡住了，已经 {int((current_time - self._last_progress_time)/60)} 分钟没有完成任何回合")
                    print("尝试自动恢复...")
                    # 重置超时计数器
                    self._last_progress_time = current_time
                    return True  # 继续训练，让环境自行处理

            # 定期保存模型，避免训练丢失
            if (self.num_timesteps - self._last_save_time_steps) >= self.save_freq:
                self._last_save_time_steps = self.num_timesteps
                self._save_model()

            # 定期报告进度
            if self.num_timesteps % self.log_interval == 0:
                self._report_progress()

            return True

        except Exception as e:
            print(f"训练进度回调出错: {e}")
            import traceback
            traceback.print_exc()
            return True


    def _save_model(self) -> None:
        """ Manual save override to print save info """
        if self.save_path is not None:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {path}")
            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                self.model.save_replay_buffer(os.path.join(self.save_path, "replay_buffer.pkl"))
                if self.verbose > 0:
                    print("Saving replay buffer")
                if self.save_vecnormalize and hasattr(self.model.get_vec_normalize_env(), "save"):
                    self.model.get_vec_normalize_env().save(os.path.join(self.save_path, "vecnormalize.pkl"))
                    if self.verbose > 0:
                        print("Saving VecNormalize statistics")


    def _report_progress(self):
        """输出训练进度报告"""
        try:
            elapsed_time = max(1e-6, time.time() - self.start_time)
            # FPS is total timesteps / elapsed time
            fps = int(self.num_timesteps / elapsed_time)

            recent_count = min(50, len(self.episode_rewards))
            if recent_count > 0:
                recent_rewards = self.episode_rewards[-recent_count:]
                recent_lengths = self.episode_lengths[-recent_count:]
                recent_exchanges = self.episode_exchange_counts[-recent_count:]

                avg_reward = sum(recent_rewards) / recent_count
                avg_length = sum(recent_lengths) / recent_count
                avg_exchanges = sum(recent_exchanges) / recent_count

                total_steps = self.num_timesteps # Use total timesteps from VecEnv
                total_desired = self.model._total_timesteps if hasattr(self.model, '_total_timesteps') else 0
                progress = (total_steps / total_desired) * 100 if total_desired > 0 else 0

                print(f"\n===== 训练进度: {total_steps}/{total_desired} 步 ({progress:.1f}%) =====")
                print(f"FPS: {fps} | 用时: {int(elapsed_time // 3600)}时{int((elapsed_time % 3600) // 60)}分{int(elapsed_time % 60)}秒")
                print(f"完成回合数: {self.total_episodes} | 最近 {recent_count} 回合平均奖励: {avg_reward:.2f}")
                print(f"平均回合长度: {avg_length:.1f} | 平均交换次数: {avg_exchanges:.2f}") # 修改打印内容
            else:
                total_steps = self.num_timesteps # Use total timesteps from VecEnv
                total_desired = self.model._total_timesteps if hasattr(self.model, '_total_timesteps') else 0
                progress = (total_steps / total_desired) * 100 if total_desired > 0 else 0

                print(f"\n===== 训练进度: {total_steps}/{total_desired} 步 ({progress:.1f}%) =====")
                print(f"FPS: {fps} | 用时: {int(elapsed_time // 3600)}时{int((elapsed_time % 3600) // 60)}分{int(elapsed_time % 60)}秒")
                print(f"尚未完成任何回合 (完成 {self.total_episodes} 个)") # 修改打印内容

        except Exception as e:
            print(f"生成进度报告时出错: {e}")
            import traceback
            traceback.print_exc()


def make_env(rank: int, seed: int = 0, render_idx: int = -1) -> callable:
    """创建环境工厂函数"""
    def _init() -> gym.Env:
        try:
            render_mode = "human" if rank == render_idx else None
            # 创建环境
            env = RotateBallsEnv(xml_file='rm_75_6f_description.xml', render_mode=render_mode)
            env.seed(seed + rank)
            
            # 将环境包装到Monitor中，收集统计信息
            log_path = f"./logs/rotate_balls_{rank}"
            os.makedirs(log_path, exist_ok=True)
            env = Monitor(env, log_path)
            
            print(f"环境 {rank} 创建成功，随机种子: {seed + rank}")
            return env
        except Exception as e:
            print(f"创建环境 {rank} 时出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果环境创建失败，返回一个简单的替代环境
            dummy_env = gym.make("CartPole-v1")
            print(f"已为环境 {rank} 创建替代环境 CartPole-v1")
            return Monitor(dummy_env, f"./logs/dummy_{rank}")

    return _init

class VecEnvSafeClose(SubprocVecEnv):
    """增强版SubprocVecEnv，确保安全关闭"""
    def close(self):
        """安全关闭所有环境"""
        if self.closed:
            return
        
        try:
            # 尝试正常关闭
            super().close()
        except Exception as e:
            print(f"安全关闭向量环境时遇到问题: {e}")
            # 强制结束子进程
            for process in self.processes:
                try:
                    if process.is_alive():
                        process.terminate()
                except Exception as proc_err:
                    print(f"终止进程时出错: {proc_err}")
        finally:
            self.closed = True


def main():
    """主函数""" 
    parser = argparse.ArgumentParser(description="机械手盘核桃 - 两球交换位置的强化学习训练")
    parser.add_argument('--total-steps', type=int, default=10000000, help='总训练步数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--eval-freq', type=int, default=10000, help='评估频率')
    parser.add_argument('--gamma', type=float, default=0.995, help='折扣因子')
    parser.add_argument('--device', type=str, default='auto', help='设备(auto/cuda/cpu)')
    parser.add_argument('--log-dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-envs', type=int, default=32, help='并行环境数量')
    parser.add_argument('--render-train', action='store_true', help='是否在训练过程中渲染环境')
    parser.add_argument('--normalize-env', action='store_true', help='是否对环境进行归一化')
    args = parser.parse_args()


    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    if device.type == 'cuda':
        try:
            print(f"GPU信息: {torch.cuda.get_device_name(0)}")
            print(f"可用GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"当前GPU内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        except Exception as e:
            print(f"获取GPU信息时出错: {e}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建环境
    if args.num_envs > 1:
        print(f"创建 {args.num_envs} 个并行环境...")
        # 如果要在训练过程中渲染，则将第一个环境设置为渲染模式
        render_idx = 0 if args.render_train else -1
        
        # 创建多个环境
        env_fns = [make_env(i, args.seed + i, render_idx) for i in range(args.num_envs)]
        
        # 使用增强版的SubprocVecEnv并行运行环境
        vec_env = VecEnvSafeClose(env_fns)
        
        # 是否对环境进行观察和奖励归一化
        if args.normalize_env:
            print("启用环境归一化...")
            # 注意：对于盘核桃任务，关闭奖励归一化，因为奖励已经有明确的意义
            vec_env = VecNormalize(
                vec_env, 
                norm_obs=True, 
                norm_reward=False,  # 关闭奖励归一化
                clip_obs=10.0, 
                clip_reward=10.0, 
                gamma=args.gamma
            )
            
        env = vec_env
        print(f"并行环境创建完成，使用 {args.num_envs} 个进程")
    else:
        print("创建单个环境...")
        # 单个环境
        render_mode = "human" if args.render_train else None
        env = RotateBallsEnv(xml_file='rm_75_6f_description.xml', render_mode=render_mode)
        # 设置种子
        env.seed(args.seed)
        # Monitor for single env
        env = Monitor(env, os.path.join(args.log_dir, "monitor"))
        # 如果使用单个环境并启用归一化
        if args.normalize_env:
             print("启用环境归一化 (单个环境)...")
             env = VecNormalize(
                DummyVecEnv([lambda: env]), 
                norm_obs=True, 
                norm_reward=False,  # 关闭奖励归一化
                clip_obs=10.0, 
                clip_reward=10.0, 
                gamma=args.gamma
             )


    # 网络架构参数 - 增强网络结构
    policy_kwargs = {
        'net_arch': [dict(pi=[512, 256, 128, 64], vf=[512, 256, 128, 64])],  # 更深的网络
        'activation_fn': nn.ReLU
    }

    # PPO算法参数
    # 根据并行环境数调整 n_steps
    n_steps_per_env = 512  # 增加步数，收集更多经验
    n_steps = n_steps_per_env

    # batch_size - PPO 更新时使用的样本数
    batch_size = 128  # 增加批次大小

    # 确保 batch_size 小于等于总的收集步数 (n_steps * num_envs)
    if batch_size > n_steps * args.num_envs:
        print(f"警告: batch_size ({batch_size}) 大于总收集步数 ({n_steps * args.num_envs}), 已调整 batch_size")
        batch_size = n_steps * args.num_envs

    # 调整batch_size为num_envs的倍数
    batch_size = (batch_size // args.num_envs) * args.num_envs

    print(f"训练配置: n_steps={n_steps}, batch_size={batch_size}, num_envs={args.num_envs}")
    print(f"每次更新收集样本数: {n_steps * args.num_envs}, 每次SGD更新样本数: {batch_size}")

    # 创建自定义回调以检测训练卡住
    class StuckDetectionCallback(BaseCallback):
        def __init__(self, check_freq=1000, patience=5, verbose=1):
            super(StuckDetectionCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.patience = patience
            self.last_reward = -float('inf')
            self.last_progress_time = time.time()
            self.stuck_count = 0
            
        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:
                # 获取平均奖励
                if len(self.model.ep_info_buffer) > 0:
                    rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                    avg_reward = np.mean(rewards) if rewards else -float('inf')
                    
                    # 重置卡住计数器
                    self.last_progress_time = time.time()
                    
                    # 检查奖励是否有明显改善
                    if avg_reward > self.last_reward * 1.01:  # 1%的改善
                        self.stuck_count = 0
                        if self.verbose:
                            print(f"奖励改善: {self.last_reward:.2f} -> {avg_reward:.2f}")
                    else:
                        self.stuck_count += 1
                        if self.verbose:
                            print(f"奖励未改善: {self.last_reward:.2f} -> {avg_reward:.2f}, 卡住计数: {self.stuck_count}/{self.patience}")
                    
                    self.last_reward = avg_reward
                    
                    # 如果连续多次没有改善，尝试修复
                    if self.stuck_count >= self.patience:
                        print("\n检测到训练卡住，尝试调整学习率...")
                        # 尝试调整学习率
                        current_lr = self.model.learning_rate
                        new_lr = current_lr * 0.5  # 减半学习率
                        if self.verbose:
                            print(f"调整学习率: {current_lr:.6f} -> {new_lr:.6f}")
                        self.model.learning_rate = new_lr
                        self.stuck_count = 0  # 重置卡住计数
                
                # 检查长时间没有进展
                current_time = time.time()
                if current_time - self.last_progress_time > 300:  # 5分钟没有进展
                    print("\n警告: 长时间没有收到回合完成信息，训练可能卡住")
                    self.last_progress_time = current_time
            
            return True

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'checkpoints'), exist_ok=True)
    
    # 设置多进程启动方法 (对Windows系统很重要)
    if sys.platform == 'win32':
        import multiprocessing
        # Windows平台必须使用spawn，避免重复初始化
        multiprocessing.set_start_method('spawn', force=True)
        print("Windows平台: 设置多进程启动方式为'spawn'")

    # PPO算法参数 - 针对精细控制任务优化
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,  # 降低学习率，提高稳定性
        n_steps=n_steps, 
        batch_size=batch_size, 
        gamma=0.995,  # 提高折扣因子，重视长期奖励
        gae_lambda=0.98,  # 提高GAE lambda
        n_epochs=15,  # 适度减少epoch数
        clip_range=0.15,  # 减小裁剪范围，更保守的更新
        clip_range_vf=0.15,
        ent_coef=0.02,  # 降低探索系数，更专注于学到的策略
        vf_coef=0.5, 
        max_grad_norm=0.3,  # 更严格的梯度裁剪
        use_sde=False, # 关闭SDE，使训练更稳定
        target_kl=0.02,  # 降低KL散度阈值
        tensorboard_log=args.log_dir,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )

    # 使用回调函数
    # 创建评估回调
    eval_seed = args.seed + 1000
    eval_env = RotateBallsEnv()
    eval_env.seed(eval_seed)
    eval_env = Monitor(eval_env, os.path.join(args.log_dir, "eval"))
    
    # 如果训练环境使用了VecNormalize，评估环境也应该加载相同的归一化统计
    if args.normalize_env:
        print("注意: 评估环境将在每次评估前加载训练环境的归一化统计")

    # 调整评估频率，确保在训练的关键阶段评估
    early_eval_freq = min(10000, args.eval_freq)
    
    # 创建定制评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, 'best_model'),
        log_path=os.path.join(args.log_dir, 'results'),
        eval_freq=early_eval_freq,
        n_eval_episodes=3,  # 减少评估回合数
        deterministic=True,
        render=False
    )

    # 创建进度回调
    progress_callback = TrainingProgressCallback(
        save_freq=10000,  # 频繁保存，避免训练丢失
        save_path=os.path.join(args.log_dir, 'checkpoints'),
        name_prefix=f'rotate_balls',
        log_interval=2000,  # 增加日志频率
        num_envs=args.num_envs,
        save_replay_buffer=False,
        save_vecnormalize=args.normalize_env,
    )
    
    # 创建卡住检测回调
    stuck_callback = StuckDetectionCallback(check_freq=1000, patience=3, verbose=1)
    
    # 组合所有回调
    callbacks = [eval_callback, progress_callback, stuck_callback]

    try:
        print("开始训练...")
        # 设置线程优先级 (在Windows上)

        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="ppo_training"
        )
        print("训练完成！")

        model.save(os.path.join(args.log_dir, "final_model"))
        # Check if env is wrapped by VecNormalize before saving its state
        if isinstance(env, VecNormalize):
            env.save(os.path.join(args.log_dir, "vecnormalize.pkl"))
            print(f"VecNormalize 状态已保存到: {os.path.join(args.log_dir, 'vecnormalize.pkl')}")
        else:
            print("训练环境未被 VecNormalize 包装，跳过保存归一化参数。")

        print(f"模型已保存到: {os.path.join(args.log_dir, 'final_model')}")

    except KeyboardInterrupt:
        print("训练被中断")
        model.save(os.path.join(args.log_dir, "interrupted_model"))
        if isinstance(env, VecNormalize):
             env.save(os.path.join(args.log_dir, "vecnormalize_interrupted.pkl"))
             print(f"中断时 VecNormalize 状态已保存到: {os.path.join(args.log_dir, 'vecnormalize_interrupted.pkl')}")
        else:
             print("训练环境未被 VecNormalize 包装，跳过保存归一化参数。")

        print(f"中断时的模型已保存到: {os.path.join(args.log_dir, 'interrupted_model')}")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        # 发生错误也保存模型
        try:
            model.save(os.path.join(args.log_dir, "error_model"))
            print(f"错误发生时的模型已保存到: {os.path.join(args.log_dir, 'error_model')}")
        except Exception as save_error:
            print(f"保存错误模型时出错: {save_error}")

    # 关闭环境
    env.close()
    if 'eval_env' in locals() and eval_env is not None:
        try:
            eval_env.close()
        except Exception as e:
            print(f"关闭评估环境时出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
