import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

plt.rcParams['axes.unicode_minus'] = False


def extract_label_from_content(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            attack_config = data.get("attack_config", {})
            if attack_config.get("attack_method", "") == "none":
                return "none"

            history = data.get("training_history", [])
            for entry in history:
                attack_method = entry.get("attack_method", "none")
                if attack_method != "none":
                    return attack_method
            return "none"
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return None


def extract_key_info(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # 从 training_history 中找出攻击方法
        attack_method = "none"
        history = data.get("training_history", [])
        for entry in history:
            method = entry.get("attack_method", "none")
            if method and method != "none":
                attack_method = method
                break  # 找到非none就退出

        env = data.get("environment_info", {})
        total_clients = env.get("total_clients", "N/A")
        num_sybil_clients = env.get("num_sybil_clients", None)
        sybil_ratio = env.get("sybil_ratio", None)
        poison_ratio = env.get("poison_ratio", None)

        # max_sybil_nodes 从 history 中取最大值
        max_sybil_nodes = max((entry.get("num_sybil_nodes", 0) for entry in history), default=0)

        # 从 history 中找到第一个 flip_ratio
        flip_ratio = None
        for entry in history:
            if "flip_ratio" in entry:
                flip_ratio = entry["flip_ratio"]
                break

        # 无攻击，简单返回总客户端数和 No Attack
        if attack_method == "none":
            return f"Total Clients: {total_clients}, No Attack"

        # 拼接 label 信息，只包含 Sybil nodes 和 ratio（flip_ratio优先，没有则sybil_ratio）
        parts = []
        parts.append(f"Sybil Nodes: {max_sybil_nodes}")

        # 优先显示 flip_ratio，否则显示 sybil_ratio
        if flip_ratio is not None:
            parts.append(f"Flip Ratio: {flip_ratio:.2f}")
        elif sybil_ratio is not None:
            parts.append(f"Sybil Ratio: {sybil_ratio:.2f}")

        return ", ".join(parts)

    except Exception as e:
        print(f"⚠️ Failed to extract info from {file_path}: {e}")
        return "Unknown"


def auto_find_sybil_files(directory="."):
    all_files = [f for f in os.listdir(directory) if f.endswith(".json") and "attack_results" in f]
    label_map = {}
    none_files = []

    for filename in all_files:
        path = os.path.join(directory, filename)
        label = extract_label_from_content(path)
        if label is None:
            continue
        label = label.strip().lower()
        if label not in label_map:
            label_map[label] = []
        label_map[label].append(path)

        if label == "none":
            none_files.append(path)

    # 确保所有非无攻击类别都附带无攻击对照组
    for attack_label in list(label_map.keys()):
        if attack_label != "none" and none_files:
            for none_file in none_files:
                if none_file not in label_map[attack_label]:
                    label_map[attack_label].append(none_file)

    print(f"🔍 Found {len(all_files)} files: {all_files}")
    return label_map


def parse_accuracy_and_loss(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            rounds = []
            accuracies = []
            losses = []
            for entry in data["training_history"]:
                rounds.append(entry["round"])
                accuracies.append(entry.get("accuracy", 0))
                losses.append(entry.get("loss", 0))
            return rounds, accuracies, losses
    except Exception as e:
        print(f"⚠️ Failed to parse file {json_path}: {e}")
        return [], [], []


from collections import defaultdict

def plot_accuracy_loss_grouped(label_map):
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    for label, file_list in label_map.items():
        # 按 max_round 分组
        round_group_map = defaultdict(list)
        for path in file_list:
            rounds, _, _ = parse_accuracy_and_loss(path)
            if not rounds:
                continue
            max_round = max(rounds)
            round_group_map[max_round].append(path)

        for max_round, files_in_round in round_group_map.items():
            plt.figure(figsize=(25, 5))
            line_styles = {
                "none": {"color": "gray", "linestyle": "--", "linewidth": 2},
                "default": {"color": None, "linestyle": "-", "linewidth": 1}
            }

            # 自动决定 X 轴步长，确保坐标不拥挤（10 以内不变，10~50 每隔5，50+每隔10）
            if max_round <= 10:
                step = 1
            elif max_round <= 50:
                step = 5
            else:
                step = 10

            # Accuracy subplot
            plt.subplot(1, 2, 1)
            for path in files_in_round:
                rounds, accs, _ = parse_accuracy_and_loss(path)
                legend_label = extract_key_info(path)
                style = line_styles["none"] if "none" in path.lower() else line_styles["default"]
                color = style["color"] if style["color"] else None
                plt.plot(rounds, accs, label=legend_label,
                         linestyle=style["linestyle"],
                         linewidth=style["linewidth"],
                         color=color)
                #plt.scatter(rounds, accs, s=15, color=color, alpha=0.7)

            plt.xlabel("Training Round", fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            plt.title(f"Accuracy Curve - {label.capitalize()} ({max_round} rounds)", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, max_round + 1)
            plt.xticks(np.arange(0, max_round + 1, step=step))  # ✅ 自动步长
            plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

            # Loss subplot
            plt.subplot(1, 2, 2)
            for path in files_in_round:
                rounds, _, losses = parse_accuracy_and_loss(path)
                legend_label = extract_key_info(path)
                style = line_styles["none"] if "none" in path.lower() else line_styles["default"]
                color = style["color"] if style["color"] else None
                plt.plot(rounds, losses, label=legend_label,
                         linestyle=style["linestyle"],
                         linewidth=style["linewidth"],
                         color=color)
                #plt.scatter(rounds, losses, s=15, color=color, alpha=0.7)

            plt.xlabel("Training Round", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title(f"Loss Curve - {label.capitalize()} ({max_round} rounds)", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, max_round + 1)
            plt.xticks(np.arange(0, max_round + 1, step=step))  # ✅ 自动步长
            plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{label}_round{max_round}_acc_loss.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Saved plot: {save_path}")

if __name__ == "__main__":
    label_map = auto_find_sybil_files()
    plot_accuracy_loss_grouped(label_map)
