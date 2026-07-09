"""
Kaggle 一键训练脚本（2x T4）——整段复制到一个 Notebook cell 运行即可。

流程：装依赖 -> 定位数据集 -> torchrun 拉起 DDP 训练 -> 打印 loss 监控。
只需改下面 CONFIG 区的 MODEL / DATASET（DATASET 留空则自动搜索挂载的数据集）。

为什么用 torchrun 而非 `accelerate launch`：Kaggle 的 accelerate CLI 启动时会
import timm/torchvision，版本一错就在训练前崩（operator torchvision::nms does not
exist）。torchrun 是 PyTorch 自带启动器，不碰这些；train.py 的 PartialState()
会自动读 torchrun 的环境变量。
"""

import glob
import json
import os
import subprocess
import sys

# ────────────────────────── CONFIG ──────────────────────────
MODEL = "Qwen/Qwen3-8B"     # 稳。冲 3.5 改 "Qwen/Qwen3.5-9B"（T4 上可能极慢/OOM）
DATASET = ""                # 留空自动搜索；或写死 "/kaggle/input/xxx/sft"
REPO_URL = "https://github.com/<your>/MyClone.git"   # 换成你的仓库
NPROC = 2                   # GPU 数；单卡填 1

EPOCHS = 1
LR = "3e-5"
LORA_RANK = "16"
LORA_DROPOUT = "0.05"
EVAL_RATIO = "0.05"
EVAL_STEPS = "50"
PATIENCE = "3"
OUTPUT_DIR = "/kaggle/working/output"
# ─────────────────────────────────────────────────────────────


def sh(cmd, check=True):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def step_install():
    print("=" * 60, "\n[1/4] 安装依赖\n", "=" * 60)
    # Text-only fine-tuning does not need these packages. Kaggle images often
    # have a torch/torchvision mismatch that Transformers wraps as
    # "Could not import module 'BloomPreTrainedModel'".
    sh("pip -q uninstall -y torchvision timm", check=False)
    pkgs = ('"transformers>=5.4.0" "trl>=0.20.0" "peft>=0.13.0" '
            '"bitsandbytes>=0.44.0" "accelerate>=1.0.0" "datasets>=3.0.0"')
    sh(f"pip -q install {pkgs}")
    import torch
    n = torch.cuda.device_count()
    print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
          "| GPUs", n, [torch.cuda.get_device_name(i) for i in range(n)])
    # 冲 Qwen3.5 且发布版报 not recognized 时，取消下一行注释装源码版：
    # sh("pip -q install 'git+https://github.com/huggingface/transformers.git'")


def step_code_and_data():
    print("=" * 60, "\n[2/4] 拉代码 + 定位数据集\n", "=" * 60)
    if not os.path.isdir("MyClone"):
        sh(f"git clone {REPO_URL} MyClone")
    else:
        sh("cd MyClone && git pull -q", check=False)

    ds = DATASET
    if not ds:
        hits = glob.glob("/kaggle/input/**/sft-my.json", recursive=True)
        if not hits:
            sys.exit("未找到 sft-my.json —— 请先把 dataset/ 作为 Kaggle Dataset 挂载。")
        ds = os.path.dirname(hits[0])
        print("自动定位数据集:", ds)
        if len(hits) > 1:
            print("（发现多个，用了第一个；如需指定请填 CONFIG 的 DATASET）")
    else:
        assert os.path.exists(os.path.join(ds, "sft-my.json")), f"{ds} 下无 sft-my.json"
    return ds


def step_train(dataset):
    print("=" * 60, f"\n[3/4] 训练 {MODEL}（{NPROC} GPU）\n", "=" * 60)
    launcher = f"torchrun --nproc_per_node={NPROC}" if NPROC > 1 else "python"
    cmd = (
        f"cd MyClone && {launcher} train.py "
        f"--model {MODEL} --dataset '{dataset}' --output {OUTPUT_DIR} "
        f"--epochs {EPOCHS} --lr {LR} --lora-rank {LORA_RANK} "
        f"--lora-dropout {LORA_DROPOUT} --eval-ratio {EVAL_RATIO} "
        f"--eval-steps {EVAL_STEPS} --early-stopping-patience {PATIENCE} "
        f"2>&1 | tee /kaggle/working/train.log"
    )
    sh(cmd)


def step_monitor():
    print("=" * 60, "\n[4/4] Loss 监控\n", "=" * 60)
    states = sorted(glob.glob(f"{OUTPUT_DIR}/**/trainer_state.json", recursive=True),
                    key=os.path.getmtime)
    if not states:
        print("没有 trainer_state.json（可能未触发 eval 或训练异常），"
              "看上面的 train.log 输出。")
        return
    hist = json.load(open(states[-1]))["log_history"]
    train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]

    ev = dict(evals)
    print(f"{'step':>6} | {'train':>8} | {'eval':>8}\n" + "-" * 30)
    for step, tl in train:
        el = ev.get(step)
        print(f"{step:>6} | {tl:>8.4f} | {('%.4f' % el) if el is not None else '':>8}")

    if evals:
        best_step, best = min(evals, key=lambda x: x[1])
        last = evals[-1][1]
        print(f"\n最优 eval_loss = {best:.4f} @ step {best_step}；最新 = {last:.4f}")
        if last > best * 1.03:
            print("⚠️ 最新 eval_loss 高于最优 3%+，有过拟合迹象。"
                  "load_best_model_at_end 已保留最优权重；"
                  "如不满意可把 LR 降到 2e-5 或 LORA_RANK 改 8 重训。")
        else:
            print("✅ eval_loss 在健康区间。")
    print(f"\nLoRA 适配器已保存到 {OUTPUT_DIR}/（从 Notebook 的 Output 标签下载）。")


if __name__ == "__main__":
    step_install()
    dataset = step_code_and_data()
    step_train(dataset)
    step_monitor()
