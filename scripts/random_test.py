from package.envs.env_full_obs import RobosuiteLiftWrapper
import numpy as np

env = RobosuiteLiftWrapper()
env.reset(seed=42)

positions = []
for i in range(10):
    obs, _ = env.reset()  # 不传seed
    cube_pos = obs[-7:-4]  # 提取cube位置
    positions.append(cube_pos)
    print(f"Reset {i}: cube_pos = {cube_pos}")
    # 检查是否所有位置都相同
    print("\n所有位置相同？", all(np.allclose(p, positions[0]) for p in positions))