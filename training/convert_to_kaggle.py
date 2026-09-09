"""convert_to_kaggle.py — convert a Tinker checkpoint to Kaggle-loadable format.

Reproduces notebook_tinker.py's conversion (the step we skipped!):
  - UNFUSE MoE:   experts.w1 -> 128x experts.{i}.up_proj
                  experts.w2 -> 128x experts.{i}.down_proj  (broadcast shared dim)
  - MERGE Mamba:  gate_proj + x_proj -> in_proj  (via QR+SVD, best rank-32)
  - target_modules -> explicit 9-module list (NOT all-linear)
  - keep base_model.model.model.* prefix (Kaggle harness wants this; backbone ERRORs)
  - drop empty experts.w3

Result matches v3's format (12010 unfused tensors) which LOADED at 0.81.

Usage:
    uv run python convert_to_kaggle.py step75 step100 step125
"""
from __future__ import annotations
import sys, os, re, json, zipfile, shutil, tempfile
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).parent
DOWNLOADS = HERE / "downloads"
OUT_DIR = Path(os.environ.get("SUBMISSION_DIR", HERE / "submission"))  # released: set via env
V3_ZIP = OUT_DIR / "submission_v8_7_v3_prefix.zip"  # tokenizer + reference

TARGET_MODULES = ["k_proj", "o_proj", "in_proj", "q_proj", "up_proj",
                  "v_proj", "down_proj", "out_proj", "lm_head"]
TOKENIZER_FILES = ["chat_template.jinja", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json"]


def convert_tensors(in_path: str) -> dict:
    """Reproduce notebook_tinker.py tensor conversion."""
    adapter = {}
    with safe_open(in_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            adapter[k] = f.get_tensor(k)

    bases = sorted({re.sub(r"\.lora_[AB]\.weight$", "", k) for k in adapter})

    # Identify Mamba layers (gate_proj + x_proj -> in_proj)
    mamba = {}
    for base in bases:
        for proj in ("gate_proj", "x_proj"):
            if base.endswith("." + proj):
                layer = base.rsplit("." + proj, 1)[0]
                mamba.setdefault(layer, {})[proj] = base
    mamba_bases = {b for projs in mamba.values() for b in projs.values()}

    out = {}
    for base in sorted(bases):
        lora_A = adapter[f"{base}.lora_A.weight"]
        lora_B = adapter[f"{base}.lora_B.weight"]

        # skip empty w3
        if ".experts.w3" in base and lora_A.numel() == 0:
            continue
        # skip mamba projs (handled below)
        if base in mamba_bases:
            continue

        # unfuse MoE experts.w1/w2
        if ".experts.w1" in base or ".experts.w2" in base:
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()
            n_exp = lora_A.shape[0]
            proj = "up_proj" if ".w1" in base else "down_proj"
            for i in range(n_exp):
                nb = re.sub(r"\.experts\.w[12]", f".experts.{i}.{proj}", base)
                out[f"{nb}.lora_A.weight"] = lora_A[i].contiguous()
                out[f"{nb}.lora_B.weight"] = lora_B[i].contiguous()
            continue

        # direct copy
        out[f"{base}.lora_A.weight"] = lora_A
        out[f"{base}.lora_B.weight"] = lora_B

    # Mamba merge: gate_proj + x_proj -> in_proj via QR+SVD (rank-32)
    # CRITICAL: the model's in_proj output dim is 10304 (gate 4096 + x 4096 +
    # B/C/dt 2112), NOT gate+x=8192. notebook_tinker.py reads this from the base
    # model (model_key_shapes); we hardcode the proven value (v3's in_proj.lora_B
    # is (10304, 32) and v3 LOADED at 0.81). The extra 2112 rows (state-space
    # B/C/dt projections) stay zero — the LoRA simply doesn't touch them.
    IN_PROJ_DIM = 10304
    for layer, projs in sorted(mamba.items()):
        in_base = f"{layer}.in_proj"
        gate_A = adapter[f"{projs['gate_proj']}.lora_A.weight"].float()
        gate_B = adapter[f"{projs['gate_proj']}.lora_B.weight"].float()
        x_A = adapter[f"{projs['x_proj']}.lora_A.weight"].float()
        x_B = adapter[f"{projs['x_proj']}.lora_B.weight"].float()
        rank = gate_A.shape[0]
        in_dim = IN_PROJ_DIM

        A_cat = torch.cat([gate_A, x_A], dim=0)            # (2r, din)
        B_block = torch.zeros(in_dim, 2 * rank)
        B_block[:gate_B.shape[0], :rank] = gate_B
        B_block[gate_B.shape[0]:gate_B.shape[0] + x_B.shape[0], rank:] = x_B

        Q_B, R_B = torch.linalg.qr(B_block)
        Q_A, R_A = torch.linalg.qr(A_cat.T)
        U, S, Vh = torch.linalg.svd(R_B @ R_A.T, full_matrices=False)
        k = rank
        new_B = (Q_B @ U[:, :k]) * S[:k].unsqueeze(0)
        new_A = Vh[:k, :] @ Q_A.T
        out[f"{in_base}.lora_A.weight"] = new_A.to(torch.float32).contiguous()
        out[f"{in_base}.lora_B.weight"] = new_B.to(torch.float32).contiguous()

    return out


def package(name: str):
    src = DOWNLOADS / name / "adapter_model.safetensors"
    if not src.exists():
        print(f"[skip] {name}: missing"); return
    print(f"\n=== {name} ===")
    tensors = convert_tensors(str(src))
    print(f"  converted -> {len(tensors)} tensors (v3 had 12010)")
    # key format sanity
    bb = sum(1 for k in tensors if ".backbone." in k)
    mm = sum(1 for k in tensors if "base_model.model.model." in k)
    fused = sum(1 for k in tensors if ".experts.w" in k)
    inp = sum(1 for k in tensors if ".in_proj." in k)
    print(f"  keyfmt: backbone={bb} model.model={mm} | fused_remain={fused} in_proj={inp}")

    work = Path(tempfile.mkdtemp(dir=str(OUT_DIR)))
    try:
        save_file(tensors, str(work / "adapter_model.safetensors"))
        # config with explicit target_modules (read base from checkpoint, override)
        cfg = json.load(open(DOWNLOADS / name / "adapter_config.json"))
        cfg["target_modules"] = TARGET_MODULES
        cfg["inference_mode"] = True
        json.dump(cfg, open(work / "adapter_config.json", "w"), indent=2)
        # tokenizer from v3
        with zipfile.ZipFile(V3_ZIP) as zv:
            for fn in TOKENIZER_FILES:
                if fn in zv.namelist():
                    (work / fn).write_bytes(zv.read(fn))

        out_zip = OUT_DIR / f"submission_v9_{name}_unfused.zip"
        if out_zip.exists():
            out_zip.unlink()
        members = ["adapter_config.json", "adapter_model.safetensors"] + TOKENIZER_FILES
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for fn in members:
                if (work / fn).exists():
                    z.write(work / fn, arcname=fn)
        print(f"  [ok] {out_zip.name} ({out_zip.stat().st_size/1e6:.0f} MB)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    names = sys.argv[1:] or ["step75", "step100", "step125"]
    for n in names:
        package(n)
    print("\n[done]")


if __name__ == "__main__":
    main()
