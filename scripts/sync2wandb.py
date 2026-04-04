import os
from collections import defaultdict

import wandb
from tensorboard.backend.event_processing import event_accumulator

ENTITY = "zixinzhao-shandong-university"
PROJECT = "state_full_obs"
BASE_DIR = "./runs"

# TensorBoard 本身已按固定间隔记录，无需脚本再次采样
# 若仍需在此基础上稀疏，可设为 >1 的整数（单位与 TB 记录间隔一致）
LOG_EVERY_N_STEPS = 1

# 若网络访问 storage.googleapis.com 卡顿，可在命令行 export WANDB_MODE=offline
WANDB_MODE = os.getenv("WANDB_MODE", "offline")

os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_DISABLE_GIT", "true")

# control 目录名 -> 标准名称映射
CONTROL_MAP = {
    "cart": "osc",
    "cartesian": "osc",
    "osc": "osc",
    "joint": "joint",
}


def collect_existing_run_names(api, entity, project):
    """拉取云端已有 run 名称，避免重复上传。"""
    names = set()
    try:
        for r in api.runs(f"{entity}/{project}"):
            names.add(r.name)
        print(f"[INFO] 已从 W&B 加载 {len(names)} 个已有 run 名称。")
    except Exception as exc:
        print(f"[WARN] 无法获取 W&B 已有 run 列表: {exc}")
    return names


def find_event_dirs(tb_root):
    """递归找到包含 TensorBoard event 文件的目录。"""
    event_dirs = []
    for root, _, files in os.walk(tb_root):
        if any(f.startswith("events.out.tfevents") for f in files):
            event_dirs.append(root)
    return event_dirs


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


def load_scalars(event_dirs, log_every_n):
    """
    读取所有 event 目录中的 scalar 数据，按步数采样。
    返回 dict: {step: {tag: value}}
    """
    metrics_by_step = defaultdict(dict)

    for event_dir in event_dirs:
        print(f"  读取: {event_dir}")
        ea = event_accumulator.EventAccumulator(event_dir)
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        if not tags:
            print(f"  [WARN] 未找到 scalar tags: {event_dir}")
            continue

        for tag in tags:
            try:
                scalars = ea.Scalars(tag)
            except Exception as exc:
                print(f"  [WARN] 读取 tag '{tag}' 失败: {exc}")
                continue

            for i, ev in enumerate(scalars):
                if log_every_n > 1 and i % log_every_n != 0:
                    continue
                metrics_by_step[ev.step][tag] = float(ev.value)

    return metrics_by_step


def upload_run(run_name, control, algo, seed, run_time, metrics_by_step):
    """初始化一个 W&B run 并上传所有采样数据。"""
    with wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=f"{control}_{algo}",
        name=run_name,
        config={
            "control": control,
            "algo": algo,
            "seed": safe_int(seed),
            "run_time": run_time,
            "log_every_n_steps": LOG_EVERY_N_STEPS,
        },
        mode=WANDB_MODE,
    ):
        for step in sorted(metrics_by_step.keys()):
            wandb.log(metrics_by_step[step], step=step)
        wandb.finish(exit_code=0)  # 加这一行
    print(f"[OK] 上传完成: {run_name}  ({len(metrics_by_step)} 个采样步)")


def main():
    api = wandb.Api()
    existing_run_names = collect_existing_run_names(api, ENTITY, PROJECT)

    if not os.path.isdir(BASE_DIR):
        print(f"[ERROR] BASE_DIR 不存在: {BASE_DIR}")
        return

    for exp in sorted(os.listdir(BASE_DIR)):
        exp_path = os.path.join(BASE_DIR, exp)
        if not os.path.isdir(exp_path):
            continue

        parts = exp.split("_")
        if len(parts) != 3:
            print(f"[SKIP] {exp}  (期望 3 段 '_' 分隔，实际 {len(parts)} 段)")
            continue

        raw_control, algo, _ = parts
        control = CONTROL_MAP.get(raw_control, raw_control)

        for run_dir in sorted(os.listdir(exp_path)):
            run_path = os.path.join(exp_path, run_dir)
            if not os.path.isdir(run_path):
                continue

            # 期望格式：seed-time
            run_parts = run_dir.split("-", 1)
            if len(run_parts) != 2:
                print(f"[SKIP] {run_dir}  (不是 'seed-time' 格式)")
                continue

            seed, run_time = run_parts
            tb_path = os.path.join(run_path, "tensorboard")
            if not os.path.exists(tb_path):
                print(f"[SKIP] {run_dir}  (无 tensorboard 目录)")
                continue

            run_name = f"{control}_{algo}_seed_{seed}"

            if run_name in existing_run_names:
                print(f"[SKIP] 已存在: {run_name}")
                continue

            event_dirs = find_event_dirs(tb_path)
            if not event_dirs:
                print(f"[SKIP] {run_name}  (未找到 TensorBoard event 文件)")
                continue

            print(f"\n[RUN] {run_name}  (event 目录数: {len(event_dirs)})")

            try:
                metrics_by_step = load_scalars(event_dirs, LOG_EVERY_N_STEPS)

                if not metrics_by_step:
                    print(f"[SKIP] {run_name}  (采样后无数据)")
                    continue

                upload_run(run_name, control, algo, seed, run_time, metrics_by_step)
                existing_run_names.add(run_name)

            except Exception as exc:
                print(f"[ERROR] {run_name} 处理失败: {exc}")
                # 确保 wandb 状态被清理
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass
                continue


if __name__ == "__main__":
    main()