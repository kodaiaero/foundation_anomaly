import torch
import os, sys, subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_device():
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.backends.mps.is_available():
    device = "mps"
  else:
    device = "cpu"
  print(f"[INFO] Using device: {device}")
  return device

def env(key, default=None, required=False):
  v = os.getenv(key, default)
  if required and not v:
    print(f"[ERR] missing env: {key}", file=sys.stderr)
    sys.exit(1)
  return v

def gcs_uri(bucket, *parts):
  return "gs://" + bucket + "/" + "/".join(p.strip("/") for p in parts)

def sh(cmd):
  print("[SH]", " ".join(cmd))
  subprocess.check_call(cmd)

def cp_gcs(src, dst, dry=False):
  if dry:
    print(f"[DRY] cp {src} -> {dst}")
    return
  sh(["gsutil", "-m", "cp", src, dst])

def upload_text(content, uri, dry=False):
  tmp = Path(".tmp_upload.txt")
  tmp.write_text(content)
  if dry:
    print(f"[DRY] upload {uri}")
  else:
    sh(["gsutil", "cp", str(tmp), uri])
  tmp.unlink(missing_ok=True)
