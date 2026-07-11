"""Create a self-contained Kaggle input bundle from the current worktree."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/dataset/sft")
    parser.add_argument("--output", default="./dist/myclone-kaggle-bundle-v3.1.zip")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    validator = root / "validate_dataset.py"

    validation = subprocess.run(
        [sys.executable, str(validator), "--dataset", str(dataset)],
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    print(validation.stdout, end="")
    if validation.returncode:
        print(validation.stderr, file=sys.stderr, end="")
        return validation.returncode

    files = {
        root / "train.py": "MyClone/train.py",
        root / "requirements.txt": "MyClone/requirements.txt",
        root / "validate_dataset.py": "MyClone/validate_dataset.py",
        root / "README.md": "MyClone/README.md",
        root / "run_kaggle.py": "run_kaggle.py",
        dataset / "sft-my.json": "dataset/sft/sft-my.json",
        dataset / "dataset_info.json": "dataset/sft/dataset_info.json",
        dataset / "quality-report.json": "dataset/sft/quality-report.json",
    }
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        print("Missing bundle files:\n" + "\n".join(missing), file=sys.stderr)
        return 2

    manifest = {
        "bundle_version": "3.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {archive: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for path, archive in files.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for source, archive in files.items():
            bundle.write(source, archive)
        bundle.writestr("bundle_manifest.json",
                        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    with zipfile.ZipFile(output) as bundle:
        bad = bundle.testzip()
        if bad:
            print(f"Bundle CRC check failed: {bad}", file=sys.stderr)
            return 3
    print(json.dumps({
        "bundle": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "entries": len(files) + 1,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
