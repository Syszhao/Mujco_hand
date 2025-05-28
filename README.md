# 机械手旋转球体的PPO强化学习训练

本项目使用PPO（Proximal Policy Optimization）算法训练机械手控制两个球体做旋转运动。

## 环境要求

- Python 3.7+
- MuJoCo 2.1.0+
- Gymnasium
- Stable-Baselines3
- PyTorch 1.10+
- CUDA工具包（用于GPU加速，推荐10.2+）

## 安装依赖

```bash
pip install -r requirements.txt
```

安装CUDA和PyTorch：
1. 访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 选择适合您系统的安装命令
2. 对于CUDA，请访问 [NVIDIA官网](https://developer.nvidia.com/cuda-downloads) 获取安装指南

## 文件说明

- `rm_75_6f_description.xml`: MuJoCo环境模型文件，包含机械手和两个球体的物理模型
- `train_rotate_balls.py`: 训练脚本，实现了环境和PPO训练逻辑
- `test_rotate_balls.py`: 测试脚本，用于测试训练好的模型
- `requirements.txt`: 项目依赖文件

## 错误修复和稳定性改进

最新版本修复了以下问题：
- 修复了`IndexError: invalid index to scalar variable`错误
- 增强了代码的错误处理能力，防止程序崩溃
- 添加了全面的异常捕获和安全检查
- 优化了内存使用，减少了OOM（内存不足）错误的可能性

主要改进：
1. 添加了对球体索引的边界检查
2. 实现了安全的数组访问和索引操作
3. 添加了调试模式，可以显示详细的错误信息
4. 减少了并行环境数量和批量大小，降低内存使用

## 使用GPU训练

本项目现已支持使用GPU加速训练过程，代码会自动检测是否有可用的CUDA设备：

- 如果检测到GPU，将自动使用GPU进行训练（推荐）
- 如果没有可用的GPU，将回退到CPU训练（较慢）

GPU配置相关信息：
- 训练代码会自动检测并使用第一个可用的GPU
- 启用了cuDNN的自动调优，提高训练效率
- GPU模式下使用更大的批量大小和更深的网络

## 训练模型

运行以下命令开始训练：

```bash
python train_rotate_balls.py
```

训练过程会自动检测GPU可用性，并显示训练进度信息：
- 显示使用的设备信息（GPU或CPU）
- 实时显示训练进度
- 显示训练完成后的总耗时

训练过程中会：
- 创建`./ppo_rotate_balls_tensorboard/`目录，保存TensorBoard日志
- 创建`./ppo_rotate_balls_checkpoints/`目录，保存中间检查点
- 训练完成后生成`ppo_rotate_balls_final.zip`（最终模型）和`vec_normalize.pkl`（归一化参数）文件

训练参数：
- 使用16个并行环境（降低了数量以减少内存使用）
- 批量大小：GPU模式下32，CPU模式下16
- 总训练步数：100,000（可根据需要增加）
- 学习率：3e-4
- 每100,000步保存一次检查点

## 测试模型

训练完成后，可使用以下命令测试模型：

```bash
python test_rotate_balls.py --model ppo_rotate_balls_final.zip --episodes 3
```

参数说明：
- `--model`: 指定模型文件路径（默认：`ppo_rotate_balls_final.zip`）
- `--normalize`: 指定归一化参数文件路径（默认：`vec_normalize.pkl`）
- `--episodes`: 测试回合数（默认：3）
- `--cpu`: 强制使用CPU进行推断，即使有可用的GPU（可选）
- `--debug`: 启用调试模式，显示详细的错误信息（可选）

## 调试模式

如果遇到问题，可以使用调试模式获取更多信息：

```bash
python test_rotate_balls.py --model ppo_rotate_balls_final.zip --debug
```

调试模式会：
- 显示详细的错误堆栈信息
- 打印更多的警告和状态信息
- 帮助定位问题所在

## GPU训练性能提升

在具有CUDA支持的GPU上训练，相比CPU训练通常可以获得显著的性能提升：
- 训练速度提升5-20倍（取决于您的GPU型号）
- 允许使用更大的批量和网络规模
- 支持更高效的并行环境模拟

## 奖励函数设计

奖励函数主要由以下几个部分组成：

1. **旋转奖励**: 基于两个球体旋转的角速度，奖励旋转速度
2. **方向一致性奖励**: 当两个球体朝同一方向旋转时给予额外奖励
3. **高度维持惩罚**: 如果球体高度超出合理范围则惩罚
4. **距离维持惩罚**: 鼓励两个球体保持在合理距离内
5. **动作大小惩罚**: 惩罚过大的控制信号

## 观察空间

观察空间包含以下信息：
- 手部关节角度和角速度
- 两个球体的位置、速度和四元数
- 两个球体之间的相对位置
- 手部位置及其与两个球的相对向量

## 注意事项

1. GPU训练需要适当的硬件和驱动支持
   - NVIDIA GPU与最新驱动
   - 已正确安装CUDA和cuDNN
2. 确保系统有足够的GPU内存（建议至少4GB）
3. 训练过程中可以使用TensorBoard监控训练进度：
   ```bash
   tensorboard --logdir ./ppo_rotate_balls_tensorboard/
   ```
4. 如果遇到内存不足错误，尝试进一步减少并行环境数量或批量大小

## 自定义训练

如需要从检查点继续训练，请修改`train_rotate_balls.py`中的：
```python
load_from_checkpoint = True
checkpoint_path = "你的检查点路径"
```

然后重新运行训练脚本。

## 常见问题解决

1. **CUDA内存不足**：减小批量大小或减少并行环境数量
2. **训练速度慢**：确保CUDA正确安装且PyTorch已编译支持GPU
3. **GPU未被使用**：检查CUDA安装和PyTorch版本是否匹配
4. **IndexError错误**：已在最新版本中修复，如仍然出现，请使用`--debug`模式获取详细信息
5. **球体位置错误**：检查XML文件中球体的定义和初始位置 