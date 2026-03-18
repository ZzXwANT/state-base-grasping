import os
import re
import ast
import yaml
from datetime import datetime


def safe_parse_td3(value_str: str):
    """
    将 NormalActionNoise(...) 替换为带引号字符串，
    使整个 dict 能被 ast.literal_eval 解析。
    """

    pattern = r"NormalActionNoise\([^)]+\)"

    def replacer(match):
        return f'"{match.group(0)}"'  # 给整个对象加引号

    value_str = re.sub(pattern, replacer, value_str)

    return ast.literal_eval(value_str)


def parse_old_txt(file_path: str):
    """
    解析你旧的 run_config.txt 格式：
    每行 key: value
    """

    config = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # 专门处理 td3_config
            if key == "td3_config":
                try:
                    value = safe_parse_td3(value)
                except Exception as e:
                    print(f"   -> td3_config 解析失败: {e}")
                    value = value  # 保留原字符串
            else:
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    pass  # 保持字符串

            config[key] = value

    return config


def convert_runs(root_dir: str):
    print(f"[INFO] 根目录: {root_dir}")
    print("=" * 60)

    if not os.path.isdir(root_dir):
        print("[ERROR] 根目录不存在")
        return

    subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    print(f"[INFO] 发现 {len(subdirs)} 个子目录")
    print("-" * 60)

    total_success = 0
    total_failed = 0

    for folder in subdirs:
        run_dir = os.path.join(root_dir, folder)
        txt_path = os.path.join(run_dir, "run_config.txt")
        yaml_path = os.path.join(run_dir, "run_config.yaml")

        print(f"[CHECK] {folder}")

        if not os.path.exists(txt_path):
            print("   -> 未找到 run_config.txt")
            continue

        if os.path.exists(yaml_path):
            print("   -> run_config.yaml 已存在，跳过")
            continue

        try:
            config_dict = parse_old_txt(txt_path)

            output = {
                "timestamp_converted": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": config_dict,
            }

            with open(yaml_path, "w") as f:
                yaml.dump(
                    output,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )

            print("   -> 转换成功")
            total_success += 1

        except Exception as e:
            print(f"   -> 转换失败: {e}")
            total_failed += 1

        print("-" * 60)

    print("\n转换完成")
    print(f"成功: {total_success}")
    print(f"失败: {total_failed}")


if __name__ == "__main__":
    target_path = "./joint_state_base/runs/td3_lift"  # 改成你的实验根目录
    convert_runs(target_path)