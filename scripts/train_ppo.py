import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from datetime import datetime
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from package.utils.PeriodSaveCallback import PeriodicSaveCallback
from package.utils.SyncEvalCallback import SyncEvalCallback
from package.utils.config_save import save_run_config

from package.envs.env_all_obs import RobosuiteLiftWrapper


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = os.path.join("./runs", f"{args.controller}_{args.exp_name}", TIME_TAG)
    model_dir = os.path.join(base_log_dir, "models")
    tb_log_dir = os.path.join(base_log_dir, "tensorboard")

    set_global_seeds(args.seed)

    env_config = dict(
        reward_shaping=True,
        control_freq=20,
        horizon=512,
        action_penalty=args.action_penalty,
        action_smooth=args.action_smooth,
        kp=args.kp,
        reward_scale=args.reward_scale,
        controller=args.controller,
    )

    norm_train = dict(norm_obs=True, norm_reward=True,  clip_obs=10.0, training=True)
    norm_eval  = dict(norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    ppo_config = dict(
        seed=args.seed,
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        n_steps=1024,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        device=args.device,
    )

    train_env = RobosuiteLiftWrapper.make_vec_env(args.num_cpu, env_config, norm_train, args.seed)
    eval_env  = RobosuiteLiftWrapper.make_vec_env(1,         env_config, norm_eval,  args.seed + 10000)

    save_run_config(
        dict(exp=args.exp_name, 
            total_timesteps=args.total_timesteps,
            num_cpu=args.num_cpu, norm_train=norm_train, 
            env_config=env_config, ppo_config=ppo_config
        ),
            save_dir=base_log_dir,
    )

    eval_dir = os.path.join(model_dir, "eval")
    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=eval_dir,
            log_path=os.path.join(model_dir, "eval_logs"),
            eval_freq=max(1, args.eval_steps // args.num_cpu),
            deterministic=True,
            callback_on_new_best=SyncEvalCallback(save_path=eval_dir),
        ),
        PeriodicSaveCallback(
            save_freq=args.model_save_freq,
            save_dir=os.path.join(model_dir, "checkpoints"),
            name_prefix="ppo",
        ),
    ])

    model = PPO(env=train_env, tensorboard_log=tb_log_dir, **ppo_config)
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
        # model.save(os.path.join(model_dir, "final_model"))
        # train_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
        print("[INFO] Training finished.")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name",          type=str,   default="ppo_lift")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--num_cpu",           type=int,   default=6)
    p.add_argument("--total_timesteps",   type=int,   default=2_000_000)
    p.add_argument("--learning_rate",     type=float, default=3e-4)
    p.add_argument("--device",            type=str,   default="cpu")
    p.add_argument("--eval_steps",        type=int,   default=163840)
    p.add_argument("--model_save_freq",   type=int,   default=500_000)
    p.add_argument("--action_penalty",    type=float, default=0)
    p.add_argument("--action_smooth",     type=float, default=0)
    p.add_argument("--kp",                type=float, default=150.0)
    p.add_argument("--reward_scale",      type=float, default=1.0)
    p.add_argument("--controller",        type=str,   default="cart", choices=["cart", "joint"])
    main(p.parse_args())