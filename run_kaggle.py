"""
Kaggle v3 一键训练脚本（2x T4）——整段复制到一个 Notebook cell 运行即可。

流程：装依赖 -> 定位数据集 -> torchrun 拉起 DDP 训练 -> 打印 loss 监控。
只需改下面 CONFIG 区的 MODEL / DATASET（DATASET 留空则自动搜索挂载的数据集）。

单卡放得下的模型（Qwen3.5-4B、Qwen3-8B）：MODEL_PARALLEL=False，走 DDP，两卡加速。
单卡放不下的大模型（Qwen3.5-9B）在 T4 上会 OOM —— DDP 是每张卡各存一份完整模型，
加卡只增吞吐不增显存。这时把 MODEL_PARALLEL=True，改用 device_map="auto" 把一个
模型切到两张 T4（约 32GB 合池）即可装下（串行、无加速）。

为什么用 torchrun 而非 `accelerate launch`：Kaggle 的 accelerate CLI 启动时会
import timm/torchvision，版本一错就在训练前崩（operator torchvision::nms does not
exist）。torchrun 是 PyTorch 自带启动器，不碰这些；train.py 的 PartialState()
会自动读 torchrun 的环境变量。
"""

import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ────────────────────────── CONFIG ──────────────────────────
MODEL = "Qwen/Qwen3.5-9B"     # 稳。冲 3.5 改 "Qwen/Qwen3.5-9B" 并把 MODEL_PARALLEL 设 True
DATASET = ""                # 留空自动搜索；或写死 "/kaggle/input/xxx/sft"
SOURCE_DIR = ""             # 留空优先使用上传的训练包，再回退到 git clone
REPO_URL = "https://github.com/chenyingxinghen/MyClone.git"   # 换成你的仓库
NPROC = 2                   # DDP 的 GPU 数；单卡填 1。MODEL_PARALLEL=True 时忽略
# 单卡放不下的大模型（如 Qwen3.5-9B）设 True：用 device_map="auto" 把一个模型
# 切分到两张 T4（约 32GB 合池），能装下但串行跑、无加速。此时只能单进程（python），
# 不能用 torchrun。小模型（4B/8B 能塞进单卡）保持 False 走 DDP 更快。
MODEL_PARALLEL = True

EPOCHS = 1
LR = "2e-5"
LORA_RANK = "8"
LORA_ALPHA = "4"
LORA_DROPOUT = "0.08"
EVAL_RATIO = "0.08"
EVAL_STEPS = "50"
PATIENCE = "3"
MULTILINE_OVERSAMPLE = "1"
MAX_ULTRASHORT_SHARE = "0.20"
OUTPUT_DIR = "/kaggle/working/output"
# ─────────────────────────────────────────────────────────────


def sh(cmd, check=True):
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def quote(value):
    """Quote a shell argument on Kaggle/Linux and during local Windows checks."""
    value = str(value)
    return subprocess.list2cmdline([value]) if os.name == "nt" else shlex.quote(value)


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


def _safe_extract(zip_path, target):
    """Extract a bundle while rejecting paths outside the working directory."""
    target = Path(target).resolve()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if target not in destination.parents and destination != target:
                raise RuntimeError(f"Unsafe path in bundle: {member.filename}")
        bundle.extractall(target)


def _find_source_and_bundle_dataset():
    if SOURCE_DIR:
        source = Path(SOURCE_DIR).resolve()
        if not (source / "train.py").exists():
            sys.exit(f"SOURCE_DIR 下无 train.py: {source}")
        return source, None

    # Recommended path: upload prepare_kaggle_bundle.py's zip as a Kaggle Dataset.
    bundles = sorted(glob.glob(
        "/kaggle/input/**/myclone-kaggle-bundle*.zip", recursive=True))
    if bundles:
        extracted = Path("/kaggle/working/myclone-training-bundle")
        print("使用上传的训练包:", bundles[-1])
        _safe_extract(bundles[-1], extracted)
        source = extracted / "MyClone"
        bundled_dataset = extracted / "dataset" / "sft"
        return source, bundled_dataset

    # Mounted input is immutable and comes from the uploaded bundle, so prefer
    # it over /kaggle/working/MyClone, which may be a stale clone from an older
    # failed run and may not recognize the current CLI arguments.
    candidates = list(
        path.parent for path in Path("/kaggle/input").glob("**/train.py")
        if path.parent.name.lower() == "myclone"
    )
    if Path("MyClone/train.py").exists():
        candidates.append(Path("MyClone").resolve())
    if candidates:
        source = candidates[0]
        train_text = (source / "train.py").read_text("utf-8", errors="replace")
        if "--lora-alpha" not in train_text or "--eval-strategy" not in train_text:
            sys.exit(
                f"检测到旧版训练代码: {source}。请删除 /kaggle/working/MyClone "
                "或重新上传最新训练包。"
            )
        return source, None

    sh(f"git clone {quote(REPO_URL)} MyClone")
    return Path("MyClone").resolve(), None


def step_code_and_data():
    print("=" * 60, "\n[2/4] 定位当前代码 + 数据集\n", "=" * 60)
    source, bundled_dataset = _find_source_and_bundle_dataset()
    print("训练代码:", source)

    ds = DATASET
    if not ds:
        hits = []
        if bundled_dataset and (bundled_dataset / "sft-my.json").exists():
            hits.append(str(bundled_dataset / "sft-my.json"))
        hits.extend(glob.glob("/kaggle/input/**/sft-my.json", recursive=True))
        if not hits:
            sys.exit("未找到 sft-my.json —— 请先把 dataset/ 作为 Kaggle Dataset 挂载。")
        ds = os.path.dirname(hits[0])
        print("自动定位数据集:", ds)
        if len(hits) > 1:
            print("（发现多个，用了第一个；如需指定请填 CONFIG 的 DATASET）")
    else:
        assert os.path.exists(os.path.join(ds, "sft-my.json")), f"{ds} 下无 sft-my.json"
    validator = source / "validate_dataset.py"
    if validator.exists():
        subprocess.run(
            [sys.executable, str(validator), "--dataset", ds],
            check=True,
        )
    else:
        print("⚠️ 当前代码中没有 validate_dataset.py，仅依赖 train.py 内置门禁。")
    return str(source), ds


def step_train(source, dataset):
    mode = "模型并行 device_map=auto，串行" if MODEL_PARALLEL else f"DDP {NPROC} GPU"
    print("=" * 60, f"\n[3/4] 训练 {MODEL}（{mode}）\n", "=" * 60)
    # 模型并行必须单进程（每个进程都会抢占所有 GPU，多进程直接 OOM）；DDP 用 torchrun。
    if MODEL_PARALLEL:
        launcher = "python"
        mp_flag = "--model-parallel "
    else:
        launcher = f"torchrun --nproc_per_node={NPROC}" if NPROC > 1 else "python"
        mp_flag = ""
    cmd = (
        f"cd {quote(source)} && {launcher} train.py {mp_flag}"
        f"--model {quote(MODEL)} --dataset {quote(dataset)} --output {quote(OUTPUT_DIR)} "
        f"--epochs {EPOCHS} --lr {LR} --lora-rank {LORA_RANK} "
        f"--lora-alpha {LORA_ALPHA} "
        f"--lora-dropout {LORA_DROPOUT} --eval-ratio {EVAL_RATIO} "
        f"--eval-strategy chronological "
        f"--multiline-oversample {MULTILINE_OVERSAMPLE} "
        f"--max-ultrashort-share {MAX_ULTRASHORT_SHARE} "
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
    source, dataset = step_code_and_data()
    step_train(source, dataset)
    step_monitor()
