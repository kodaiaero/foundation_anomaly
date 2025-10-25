import torch
from models.dinov2_model import DinoV2Model
from core.feature_extractor import compute_normal_feature_mean
from core.scoremap import compute_scoremap, save_scoremap
from PIL import Image
import glob
import os

def main():
    # ==== 1. モデル初期化 ====
    model = DinoV2Model()

    # ==== 2. 正常画像の平均特徴作成 ====
    F_mean = compute_normal_feature_mean(model, "data/normal")
    print("[INFO] Normal feature mean computed.")

    # ==== 3. テスト画像からスコアマップ生成 ====
    test_paths = glob.glob("data/test/*.jpg") + glob.glob("data/test/*.png")
    for path in test_paths:
        img = Image.open(path).convert("RGB")
        F_test = model.extract_features(img)
        score_map = compute_scoremap(F_test, F_mean)
        
        # ==== 4. 保存 ====
        filename = os.path.basename(path).split(".")[0]
        out_path = f"outputs/scoremaps/{filename}_heatmap.png"
        save_scoremap(score_map, out_path)
        print(f"[INFO] Saved: {out_path}")

if __name__ == "__main__":
    main()
