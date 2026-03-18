import os
from stable_baselines3.common.callbacks import BaseCallback

class PeriodicSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_dir, name_prefix="model", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        # 新增：记录上一次保存的步数，初始为 0 或模型加载时的步数
        self.last_time_trigger = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        # 初始化时同步一下当前模型的总步数，防止从断点恢复时瞬间触发
        self.last_time_trigger = self.model.num_timesteps

    def _on_step(self) -> bool:
        # 改为差值判断，防止跳过
        if (self.num_timesteps - self.last_time_trigger) >= self.save_freq:
            # 更新触发步数（使用已走过的步数之和，保证间隔准确）
            self.last_time_trigger = (self.num_timesteps // self.save_freq) * self.save_freq
            
            curr_steps = self.num_timesteps // 1000
            model_path = os.path.join(self.save_dir, f"{self.name_prefix}_{curr_steps}k.zip")
            self.model.save(model_path)
            
            vec_normalize = self.model.get_vec_normalize_env()
            if vec_normalize is not None:
                stats_path = os.path.join(self.save_dir, f"vecnormalize_{curr_steps}k.pkl")
                vec_normalize.save(stats_path)
                if self.verbose:
                    print(f"[INFO] 触发周期性保存: {curr_steps}k steps")
            
        return True