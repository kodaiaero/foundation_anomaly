import argparse, random, csv, os, subprocess
from core.utils import env, gcs_uri, cp_gcs, upload_text, sh
from pathlib import Path
from collections import defaultdict

def main():
  ap = argparse.ArgumentParser(description="Freeze training set per region (auto-sampling)")
  ap.add_argument("--region", required=True, help="Region name (e.g. noto, volcano)")
  ap.add_argument("--selected", default="auto", help="File list or 'auto'")
  ap.add_argument("--version", default=os.getenv("DATA_VERSION", "v1"))
  ap.add_argument("--per-group", type=int, default=50)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--train-ratio", type=float, default=0.8)
  ap.add_argument("--val-ratio", type=float, default=0.1)
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()

  bucket = env("GCS_BUCKET", required=True)
  pfx_images = env("GCS_PREFIX_IMAGES", "images")
  pfx_metadata = env("GCS_PREFIX_METADATA", "metadata")
  pfx_splits = env("GCS_PREFIX_SPLITS", "splits")

  orig_prefix = gcs_uri(bucket, pfx_images, "original", args.region)
  train_prefix = gcs_uri(bucket, pfx_images, "training", args.region, args.version)
  manifest_uri = gcs_uri(bucket, pfx_metadata, f"manifest_{args.region}_{args.version}.csv")
  splits_dir = gcs_uri(bucket, pfx_splits, args.region, args.version)

  print(f"[REGION] {args.region}")
  print(f"[SRC] {orig_prefix}")
  print(f"[DST] {train_prefix}")

  if args.selected == "auto":
    print("[INFO] Auto listing files from GCS...")
    result = subprocess.check_output(["gsutil", "ls", f"{orig_prefix}/**.jpg"], text=True)
    rels = [
      r.replace(f"{orig_prefix}/", "").strip()
      for r in result.splitlines() if r.strip()
    ]
    print(f"[INFO] Found {len(rels)} files total.")
  else:
    with open(args.selected) as f:
      rels = [ln.strip() for ln in f if ln.strip()]


  grouped = defaultdict(list)
  for r in rels:
    ds = r.split("/")[0] if "/" in r else "unknown"
    grouped[ds].append(r)

  random.seed(args.seed)
  sampled = []
  for ds, arr in grouped.items():
    random.shuffle(arr)
    n = min(args.per_group, len(arr))
    sampled += arr[:n]
    print(f"[SAMPLE] {ds}: {n} / {len(arr)}")

  rels = sampled

  # copy original → training
  for rel in rels:
    src = f"{orig_prefix}/{rel}"
    dst = f"{train_prefix}/{rel}"
    cp_gcs(src, dst, args.dry_run)

  # manifest
  rows = []
  for rel in rels:
    ds = rel.split("/")[0] if "/" in rel else "unknown"
    stem = Path(rel).stem
    image_id = f"{args.region}__{ds}__{stem}"
    rows.append((ds, image_id, rel))

  # split by dataset（region内で分割）
  train_ids, val_ids, test_ids = [], [], []
  random.seed(args.seed)
  for ds, arr in defaultdict(list, {r[0]: [] for r in rows}).items():
    arr = [(iid, rel) for (d, iid, rel) in rows if d == ds]
    random.shuffle(arr)
    n = len(arr)
    n_tr = int(n * args.train_ratio)
    n_vl = int(n * args.val_ratio)
    train_ids += [iid for iid, _ in arr[:n_tr]]
    val_ids   += [iid for iid, _ in arr[n_tr:n_tr+n_vl]]
    test_ids  += [iid for iid, _ in arr[n_tr+n_vl:]]

  # local manifest lives under metadata/
  metadata_dir = Path("metadata")
  metadata_dir.mkdir(parents=True, exist_ok=True)
  mpath = metadata_dir / f"manifest_{args.region}_{args.version}.csv"
  with mpath.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["version","region","dataset","image_id","rel_path"])
    for ds, iid, rel in rows:
      w.writerow([args.version, args.region, ds, iid, rel])

  if not args.dry_run:
    sh(["gsutil", "cp", str(mpath), manifest_uri])
    upload_text("\n".join(train_ids)+"\n", f"{splits_dir}/train.txt")
    upload_text("\n".join(val_ids)+"\n",   f"{splits_dir}/val.txt")
    upload_text("\n".join(test_ids)+"\n",  f"{splits_dir}/test.txt")

  print(f"[OK] Region {args.region} v{args.version} frozen.")
  print(f"  manifest : {manifest_uri}")
  print(f"  splits   : {splits_dir}")
  print(f"  training : {train_prefix}")

if __name__ == "__main__":
  main()
