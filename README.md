# Dexterous-operation

## Adroit RL Baselines

这个仓库包含 4 个基于 Stable-Baselines3 的强化学习的Adroit灵巧手训练脚本，用于 `gymnasium_robotics` 中的 Adroit 任务。

目前包括以下文件：

- `adroit_door_PPO.py`
- `adroit_door_SAC.py`
- `adroit_hammer_PPO.py`
- `adroit_hammer_SAC.py`

这些脚本分别对应：

- `door` 任务 + `PPO`
- `door` 任务 + `SAC`
- `hammer` 任务 + `PPO`
- `hammer` 任务 + `SAC`

---

## 1. 文件说明

### `adroit_door_PPO.py`
使用 **PPO** 算法训练 `AdroitHandDoor-v1` 任务。

### `adroit_door_SAC.py`
使用 **SAC** 算法训练 `AdroitHandDoor-v1` 任务。

### `adroit_hammer_PPO.py`
使用 **PPO** 算法训练 `AdroitHandHammer-v1` 任务。

### `adroit_hammer_SAC.py`
使用 **SAC** 算法训练 `AdroitHandHammer-v1` 任务。

---

## 2. 这些脚本做了什么

每个脚本都包含以下几个部分：

### 环境创建
通过 `gymnasium.make(...)` 创建 Adroit 环境，并用 `Monitor` 包装，方便记录 reward 和 episode 信息。

### 向量化环境
使用 `DummyVecEnv` 构建适配 Stable-Baselines3 的训练环境。

### 训练过程
使用对应算法进行训练：

- `PPO` 脚本使用 Proximal Policy Optimization
- `SAC` 脚本使用 Soft Actor-Critic

### 评估与保存
训练过程中会定期：

- 保存 checkpoint
- 在单独的 eval 环境上评估
- 保存 best model
- 输出 TensorBoard 日志
- 最后保存 final model

---

## 3. 运行环境

建议使用 **Python 3.10** 或 **Python 3.11**。

### 依赖库
主要依赖如下：

- `gymnasium`
- `gymnasium-robotics`
- `stable-baselines3`
- `mujoco`
- `numpy`
- `torch`

---

## 4. 安装方式

建议先创建虚拟环境。

### 使用 venv
```bash
python -m venv venv
source venv/bin/activate