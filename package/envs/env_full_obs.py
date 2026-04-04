import numpy as np
import gymnasium as gym
from gymnasium import spaces

import robosuite as suite
from robosuite.environments.manipulation.lift_custom import LiftCustom
suite.environments.base.REGISTERED_ENVS["lift_custom"] = LiftCustom

from robosuite.controllers import load_composite_controller_config
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
)

class RobosuiteLiftWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_shaping=True,
        control_freq=20,
        horizon=256,
        has_offscreen_renderer=False,
        action_penalty=0,
        action_smooth=0,
        kp=150,
        reward_scale=1.0,
        hold_steps=40,
        controller="cart",
    ):
        super().__init__()

        self.controller_type = controller

        cfg = load_composite_controller_config(controller=None, robot="UR5e")
        cfg.update({"kp": kp, "kp_ori": 150, "interpolation": "min_jerk"})

        if controller == "joint":
            cfg["body_parts"]["right"]["type"] = "JOINT_POSITION"
            cfg["body_parts"]["right"]["output_max"] = 0.1
            cfg["body_parts"]["right"]["output_min"] = -0.1
            controller_cfg = cfg  
            
        elif controller == "cart":
            controller_cfg = cfg
        else:
            raise ValueError(f"未知 controller 类型: {controller}")
        
        self.rs_env = suite.make(
            env_name="lift_custom",    # ppo: Lift; td3、sac: lift_custom（把lift环境中reaching奖励*0.5）
            robots="UR5e",
            has_renderer=False,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=False,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            controller_configs=controller_cfg,
            horizon=horizon,
        )

        low  = np.asarray(self.rs_env.action_spec[0], dtype=np.float32)
        high = np.asarray(self.rs_env.action_spec[1], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.prev_action  = np.zeros_like(low, dtype=np.float32)

        obs = self._process_obs(self.rs_env.reset())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        self.action_penalty = action_penalty
        self.action_smooth  = action_smooth
        self.reward_scale   = reward_scale
        self.hold_count     = 0
        self.hold_steps     = hold_steps
    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _process_obs(self, obs):
        sim = self.rs_env.sim
        f1 = sim.data.geom_xpos[sim.model.geom_name2id("gripper0_right_left_fingerpad_collision")]
        f2 = sim.data.geom_xpos[sim.model.geom_name2id("gripper0_right_right_fingerpad_collision")]
        dist = np.array([np.linalg.norm(f1 - f2)], dtype=np.float32)

        return np.concatenate(
            [obs["robot0_joint_pos"],
            obs["robot0_eef_pos"], 
            obs["robot0_eef_quat"], 
            dist, 
            obs["cube_pos"], 
            obs["cube_quat"]]
        ).astype(np.float32)
    # ------------------------------------------------------------------
    # Gym 接口
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = self.rs_env.reset()

        self.hold_count     = 0
        self.prev_action[:] = 0.0

        # print("[DEBUG] cube_pos:", raw_obs.get("cube_pos"))

        return self._process_obs(raw_obs), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, done, info = self.rs_env.step(action)

        penalty = (
            self.action_penalty * np.sum(np.square(action[:6]))                 # 不惩罚夹爪的大动作
            + self.action_smooth * np.sum(np.square(action - self.prev_action))
        )
        self.prev_action = action.copy()
        scaled_reward = (float(reward) - penalty) * self.reward_scale

        if self.rs_env._check_success():
            self.hold_count += 1
        else:
            self.hold_count = 0

        info["is_success"] = self.hold_count >= self.hold_steps

        return self._process_obs(obs), scaled_reward, False, done, info

    def close(self):
        try:
            self.rs_env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def _make_env_fn(cls, rank, env_config, base_seed):
        def _init():
            seed = base_seed + rank
            np.random.seed(seed)   # ← 只在进程启动时 seed 一次
            env = cls(**env_config)
            env.reset()
            return env
        return _init

    @classmethod
    def make_vec_env(cls, num_cpu, env_config, VecNorm, base_seed=42):
        venv = SubprocVecEnv(
            [cls._make_env_fn(rank, env_config, base_seed) for rank in range(num_cpu)]
        )
        venv = VecMonitor(venv)
        venv = VecNormalize(
            venv,
            norm_obs=VecNorm["norm_obs"],
            norm_reward=VecNorm["norm_reward"],
            clip_obs=VecNorm["clip_obs"],
            training=VecNorm["training"],
        )
        return venv

    @classmethod
    def make_vec_env_for_test(cls, num_cpu, env_config, vecnorm_path, seeding_base=42):
        venv = DummyVecEnv(
            [cls._make_env_fn(rank, env_config, seeding_base) for rank in range(num_cpu)]
        )
        venv = VecMonitor(venv)
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training    = False
        venv.norm_reward = False
        return venv