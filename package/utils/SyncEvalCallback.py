import os
from stable_baselines3.common.callbacks import BaseCallback

class SyncEvalCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        # 当 EvalCallback 发现新高分时，会调用这里
        vec_normalize = self.model.get_vec_normalize_env()
        
        if vec_normalize is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
            # 保存与 Best Model 对应的归一化参数
            stats_path = os.path.join(self.save_path, "best_vecnormalize.pkl")
            vec_normalize.save(stats_path)
            
            if self.verbose > 0:
                print(f"检测到新 Best！同步保存归一化文件: {stats_path}")
        return True
    
# class SyncEvalCallback(BaseCallback):
#     def __init__(self, save_path: str, verbose=1):
#         super().__init__(verbose)
#         self.save_path = save_path

#     def _on_step(self) -> bool:
#         # 只要被唤醒，就获取当前步数并保存
#         step = self.num_timesteps
#         vec_normalize = self.model.get_vec_normalize_env()
        
#         if vec_normalize is not None:
#             os.makedirs(self.save_path, exist_ok=True)
            
#             # 1. 保存模型
#             self.model.save(os.path.join(self.save_path, f"model_{step}_steps"))
            
#             # 2. 保存归一化文件
#             stats_path = os.path.join(self.save_path, f"vecnormalize_{step}_steps.pkl")
#             vec_normalize.save(stats_path)
            
#             if self.verbose > 0:
#                 print(f"完成定期评估，已保存模型和归一化文件: step={step}")
#         return True
