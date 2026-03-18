import os
import re
import argparse
from pathlib import Path

import wandb
from tbparse import SummaryReader  # pip install tbparse


# ─────────────────────────── 配置 ───────────────────────────
# 修改为你的 W&B entity（用户名或团队名）
WANDB_ENTITY = "zixinzhao-shandong-university"

# 要扫描的 base 目录列表（绝对路径或相对路径均可）
BASE_DIRS = [
    "./cart_state_base",
    "./joint_state_base",
]
# ─────────────────────────────────────────────────────────────


def parse_seed(folder_name: str) -> int | None:
    """
    从 '{seed}-{YYYYMMDD}_{HHMMSS}' 格式的文件夹名中提取 seed 编号。
    不符合该格式的返回 None，调用方应跳过。
    示例匹配：0-20260312_000730 → 0
    """
    match = re.match(r"^(\d+)-(\d{8})_(\d{6})$", folder_name)
    if match:
        return int(match.group(1))
    return None


def find_tb_dir(seed_dir: Path) -> Path | None:
    """在 seed 目录下递归查找包含 events.out.* 的目录"""
    for events_file in seed_dir.rglob("events.out.*"):
        return events_file.parent
    return None


def parse_seed_runidx(folder_name: str) -> tuple[int, int] | None:
    """
    从 '{seed}_{run_idx}-{YYYYMMDD}_{HHMMSS}' 格式中提取 (seed, run_idx)。
    不符合格式返回 None。
    示例：0_1-20260312_000730 → (0, 1)
    """
    match = re.match(r"^(\d+)_(\d+)-(\d{8})_(\d{6})$", folder_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def parse_entry(folder_name: str) -> tuple[int, int | None] | None:
    """
    兼容两种格式，返回 (seed, run_idx) 或 (seed, None)：
      {seed}_{run_idx}-{YYYYMMDD}_{HHMMSS}  →  (seed, run_idx)
      {seed}-{YYYYMMDD}_{HHMMSS}            →  (seed, None)
    不匹配返回 None。
    """
    result = parse_seed_runidx(folder_name)
    if result is not None:
        return result
    seed = parse_seed(folder_name)
    if seed is not None:
        return (seed, None)
    return None
 
 
def upload_1seed_dir(
    oneseed_dir: str,
    project: str,
    group: str,
    algo: str,
    base_name: str,
    dry_run: bool = False,
):
    """
    处理额外多一层子目录的特殊情况，兼容两种命名格式：
      {seed}_{run_idx}-{YYYYMMDD}_{HHMMSS}  →  run 名 {algo}_seed{seed}_run{run_idx}
      {seed}-{YYYYMMDD}_{HHMMSS}            →  run 名 {algo}_seed{seed}
    每条记录作为独立 run 上传。
    """
    oneseed_path = Path(oneseed_dir)
    if not oneseed_path.exists():
        print(f"⚠ 目录不存在: {oneseed_dir}")
        return
 
    print(f"\n{'='*60}")
    print(f"特殊目录: {oneseed_path}")
    print(f"Base: {base_name} | Algo: {algo} | Project: {project} | Group: {group}")
    print(f"{'='*60}")
 
    total, skipped = 0, 0
 
    entries = sorted(
        [d for d in oneseed_path.iterdir() if d.is_dir() and parse_entry(d.name) is not None],
        key=lambda d: parse_entry(d.name),
    )
 
    if not entries:
        print("  ⚠ 未找到符合命名格式的子目录")
        return
 
    for entry in entries:
        seed, run_idx = parse_entry(entry.name)
        tb_dir = find_tb_dir(entry)
        run_name = f"{algo}_seed{seed}_run{run_idx}" if run_idx is not None else f"{algo}_seed{seed}"
 
        if tb_dir is None:
            print(f"  ⚠ {entry.name}: 未找到 TensorBoard 事件文件，跳过")
            skipped += 1
            continue
 
        print(f"\n{'[DRY RUN] ' if dry_run else ''}上传: {tb_dir}")
        run_idx_str = f", run_idx={run_idx}" if run_idx is not None else ""
        print(f"  → project={project}, group={group}, run={run_name}, seed={seed}{run_idx_str}")
 
        if dry_run:
            total += 1
            continue
 
        reader = SummaryReader(str(tb_dir), pivot=False, extra_columns={"wall_time", "dir_name"})
        df = reader.scalars
 
        if df.empty:
            print(f"  ⚠ 未找到 scalar 数据，跳过")
            skipped += 1
            continue
 
        config = {
                "algo": algo,
                "seed": seed,
                "base": base_name,
                "tb_dir": str(tb_dir),
            }
        if run_idx is not None:
            config["run_idx"] = run_idx
 
        run = wandb.init(
            project=project,
            entity=WANDB_ENTITY,
            group=group,
            name=run_name,
            config=config,
            reinit="finish_previous",
        )
 
        for step, step_df in df.groupby("step"):
            log_dict = {}
            for _, row in step_df.iterrows():
                tag = row["tag"].replace("/", "_")
                log_dict[tag] = row["value"]
            wandb.log(log_dict, step=int(step))
 
        run.finish()
        print(f"  ✓ 上传完成")
        total += 1
 
    print(f"\n完成！共处理 {total} 个 run，跳过 {skipped} 个")


def upload_run(
    tb_dir: Path,
    project: str,
    group: str,
    seed: int,
    algo: str,
    base_name: str,
    dry_run: bool = False,
):
    """读取单个 TensorBoard 目录并上传到 W&B"""
    run_name = f"{algo}_seed{seed}"
    print(f"\n{'[DRY RUN] ' if dry_run else ''}上传: {tb_dir}")
    print(f"  → project={project}, group={group}, run={run_name}, seed={seed}")

    if dry_run:
        return

    reader = SummaryReader(str(tb_dir), pivot=False, extra_columns={"wall_time", "dir_name"})
    df = reader.scalars

    if df.empty:
        print(f"  ⚠ 未找到 scalar 数据，跳过")
        return

    run = wandb.init(
        project=project,
        entity=WANDB_ENTITY,
        group=group,
        name=run_name,
        config={
            "algo": algo,
            "seed": seed,
            "base": base_name,
            "tb_dir": str(tb_dir),
        },
        reinit="finish_previous",
    )

    # 按 step 分组上传所有 scalar
    for step, step_df in df.groupby("step"):
        log_dict = {}
        for _, row in step_df.iterrows():
            tag = row["tag"].replace("/", "_")  # W&B 不支持 tag 中的 /
            log_dict[tag] = row["value"]
        wandb.log(log_dict, step=int(step))

    run.finish()
    print(f"  ✓ 上传完成")


def scan_and_upload(base_dirs: list[str], dry_run: bool = False):
    total = 0
    skipped = 0

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        base_name = base_path.name  # 如 cart_state_base

        runs_path = base_path / "runs"
        if not runs_path.exists():
            print(f"⚠ 未找到 runs 目录: {runs_path}，跳过")
            continue

        # 遍历算法文件夹（如 sac_lift, ppo_lift, td3_lift）
        for algo_dir in sorted(runs_path.iterdir()):
            if not algo_dir.is_dir():
                continue
            algo = algo_dir.name  # 如 sac_lift

            # W&B project 名：base_name + "_" + algo（如 cart_state_base_sac_lift）
            project = f"{base_name}_{algo}"
            # group 名：用于在 project 内把多 seed 归组
            group = f"{algo}_allseeds"

            print(f"\n{'='*60}")
            print(f"Base: {base_name} | Algo: {algo} | Project: {project}")
            print(f"{'='*60}")

            # 遍历 seed 文件夹（如 0_20260312_000730）
            # 只保留严格匹配 {seed}_{YYYYMMDD}_{HHMMSS} 格式的目录
            seed_dirs = sorted(
                [d for d in algo_dir.iterdir() if d.is_dir() and parse_seed(d.name) is not None],
                key=lambda d: parse_seed(d.name),
            )

            if not seed_dirs:
                print(f"  ⚠ 未找到 seed 目录，跳过")
                continue

            for seed_dir in seed_dirs:
                seed = parse_seed(seed_dir.name)  # 已过滤，此处必不为 None
                tb_dir = find_tb_dir(seed_dir)

                if tb_dir is None:
                    print(f"  ⚠ {seed_dir.name}: 未找到 TensorBoard 事件文件，跳过")
                    skipped += 1
                    continue

                upload_run(
                    tb_dir=tb_dir,
                    project=project,
                    group=group,
                    seed=seed,
                    algo=algo,
                    base_name=base_name,
                    dry_run=dry_run,
                )
                total += 1

    print(f"\n{'='*60}")
    print(f"完成！共处理 {total} 个 run，跳过 {skipped} 个")


if __name__ == "__main__":
    # 在脚本末尾的 __main__ 块里，或单独调用：
    upload_1seed_dir(
        oneseed_dir="./cart_state_base/runs/ppo_lift/1seed",
        project="cart_state_base_ppo_lift",
        group="ppo_lift_allseeds",
        algo="ppo_lift",
        base_name="cart_state_base",
        dry_run=False,  # 先 dry run 确认
    )
    
    # parser = argparse.ArgumentParser(description="上传 TensorBoard 日志到 W&B")
    # parser.add_argument(
    #     "--dry-run",
    #     action="store_true",
    #     help="只扫描目录并打印将要上传的内容，不实际上传",
    # )
    # parser.add_argument(
    #     "--base-dirs",
    #     nargs="+",
    #     default=None,
    #     help="覆盖脚本中的 BASE_DIRS，传入一个或多个路径",
    # )
    # parser.add_argument(
    #     "--entity",
    #     default=None,
    #     help="覆盖脚本中的 WANDB_ENTITY",
    # )
    # args = parser.parse_args()

    # if args.entity:
    #     WANDB_ENTITY = args.entity
    # if args.base_dirs:
    #     BASE_DIRS = args.base_dirs

    # scan_and_upload(BASE_DIRS, dry_run=args.dry_run)