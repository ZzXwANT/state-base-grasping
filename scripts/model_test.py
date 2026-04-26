import os
import cv2
import csv
import time
import imageio
import numpy as np
from stable_baselines3 import PPO, TD3, SAC

from package.envs.env_full_obs import RobosuiteLiftWrapper

def run_tests(
    vecnorm_path,
    model_path,
    out_dir,
    num_seeds=5,
    seeds=None,
    max_steps=256,
    fps=20,
    controller=None,
    alg=None,
):
    os.makedirs(out_dir, exist_ok=True)

    if seeds is None:
        seeds = list(map(int, np.random.default_rng().integers(0, 10000, size=num_seeds)))

    env_config = {"has_offscreen_renderer": True, 
                  "horizon": max_steps,
                  "controller": controller,
                  }
    log_path = os.path.join(out_dir, "test_log.csv")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["seed", "success", "steps", "total_reward", "video_path", "duration_s"])

    for seed in seeds:
        t0 = time.time()
        print(f"\n=== Seed {seed} ===")

        venv = RobosuiteLiftWrapper.make_vec_env_for_test(
            num_cpu=1, env_config=env_config, vecnorm_path=vecnorm_path, seeding_base=seed,
        )
        model = alg.load(model_path, env=venv)
        robosim = venv.envs[0].rs_env

        tmp_path   = os.path.join(out_dir, f"_tmp_{seed}.mp4")
        vid_writer = imageio.get_writer(tmp_path, fps=fps)

        obs = venv.reset()
        done, steps, total_reward, success = False, 0, 0.0, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)

            total_reward += float(rewards[0])
            steps += 1

            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            success = bool(info0.get("is_success", False)) if isinstance(info0, dict) else False
            done = bool(dones[0]) or success

            try:
                frame = robosim.sim.render(height=512, width=512, camera_name="agentview")[::-1].copy()
                # 在右上角写 total_reward
                text = f"Reward: {total_reward:.2f}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2

                # 先计算文字大小，方便右对齐
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

                x = frame.shape[1] - text_w - 10
                y = text_h + 10

                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                vid_writer.append_data(frame)

            except Exception as e:
                print("Render failed:", e)

        vid_writer.close()

        final_path = os.path.join(out_dir, f"{total_reward:.2f}_test_{seed}.mp4")
        os.replace(tmp_path, final_path)

        duration = time.time() - t0
        print(f"success={success}, steps={steps}, reward={total_reward:.2f}, video={final_path}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([seed, int(success), steps, f"{total_reward:.4f}", final_path, f"{duration:.2f}"])

        try:
            venv.close()
        except Exception:
            pass

    print(f"\nDone. Log: {log_path}")


if __name__ == "__main__":
    controller = "joint"
    alg = PPO 
    BASE_DIR    = f"./runs/{controller}_{alg.__name__.lower()}_lift/0-20260414_135133"
    MODEL_PATH  = os.path.join(BASE_DIR, "models/eval/best_model.zip")
    NORMAL_PATH = os.path.join(BASE_DIR, "models/eval/best_vecnormalize.pkl")
    OUT_DIR     = os.path.join(BASE_DIR, "test_videos")

    assert os.path.exists(MODEL_PATH),  f"Model not found: {MODEL_PATH}"
    assert os.path.exists(NORMAL_PATH), f"VecNormalize not found: {NORMAL_PATH}"

    run_tests(NORMAL_PATH, MODEL_PATH, OUT_DIR, 
              num_seeds=5, max_steps=256, fps=20, controller=controller, alg=alg)