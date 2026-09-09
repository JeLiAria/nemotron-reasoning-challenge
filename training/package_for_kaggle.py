"""package_for_kaggle.py — wrap a Tinker checkpoint into a Kaggle submission zip.

The Tinker checkpoint is ALREADY in Tong's exact format:
  - 418 tensors, fused MoE (experts.w1/w2), gate_proj+x_proj, out_proj alive
  - keyfmt: base_model.model.model.* (0 backbone, 0 .default)
  - adapter_config.json identical to Tong's (target_modules=all-linear)

So packaging is trivial: take the checkpoint's safetensors + config, add the
tokenizer files from the PROVEN-LOAD v3 zip (byte-identical, loaded at 0.81),
zip with ZIP_STORED. No tensor surgery, no key renames.

Usage:
    uv run python package_for_kaggle.py step75 step100 step125
"""
from __future__ import annotations
import sys, os, zipfile, shutil, tempfile, hashlib
from pathlib import Path

HERE = Path(__file__).parent
DOWNLOADS = HERE / "downloads"
OUT_DIR = Path(os.environ.get("SUBMISSION_DIR", HERE / "submission"))  # released: set via env
V3_ZIP = OUT_DIR / "submission_v8_7_v3_prefix.zip"  # proven-load tokenizer source

TOKENIZER_FILES = ["chat_template.jinja", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json"]


def sha(b): return hashlib.sha256(b).hexdigest()[:12]


def package(name: str):
    ckpt = DOWNLOADS / name
    st = ckpt / "adapter_model.safetensors"
    cfg = ckpt / "adapter_config.json"
    if not st.exists():
        print(f"[skip] {name}: no adapter_model.safetensors"); return None
    out_zip = OUT_DIR / f"submission_v9_{name}.zip"

    work = Path(tempfile.mkdtemp(dir=str(OUT_DIR)))
    try:
        # 1. adapter weights + config (from Tinker, already Tong-format)
        shutil.copy(st, work / "adapter_model.safetensors")
        shutil.copy(cfg, work / "adapter_config.json")

        # 2. tokenizer files from the proven v3 zip (byte-identical)
        with zipfile.ZipFile(V3_ZIP) as zv:
            names = set(zv.namelist())
            for fn in TOKENIZER_FILES:
                if fn in names:
                    (work / fn).write_bytes(zv.read(fn))
                else:
                    print(f"    WARN: {fn} not in v3 zip")

        # 3. build zip, ZIP_STORED, 6 members
        members = ["adapter_config.json", "adapter_model.safetensors",
                   "chat_template.jinja", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json"]
        if out_zip.exists():
            out_zip.unlink()
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for fn in members:
                fp = work / fn
                if fp.exists():
                    z.write(fp, arcname=fn)
        size = out_zip.stat().st_size / 1e6
        print(f"[ok] {out_zip.name}  ({size:.0f} MB)")
        return out_zip
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    names = sys.argv[1:] or ["step75", "step100", "step125"]
    print(f"Packaging: {names}\n")
    for name in names:
        package(name)
    print(f"\n[done] submit the zips from {OUT_DIR}")


if __name__ == "__main__":
    main()
