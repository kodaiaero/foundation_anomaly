from models.dinov2_model import DinoV2Model
from core.feature_extractor import compute_normal_feature_mean
from core.scoremap import compute_scoremap, save_scoremap
from PIL import Image
import glob
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    # ==== 1. モデル初期化 ====
    model = DinoV2Model()

    # ==== 2. 正常画像の平均特徴作成 ====
    category = os.getenv("NORMAL_CATEGORY", "normal") # デフォルトは'normal'
    normal_data_dir = f"data/{category}"
    F_mean = compute_normal_feature_mean(model, normal_data_dir)
    print(f"[INFO] Normal feature mean computed for category: {category}")

    # ==== 3. テスト画像からスコアマップ生成 ====
    test_paths = glob.glob("data/test/*.jpg") + glob.glob("data/test/*.png")
    for path in test_paths:
        img = Image.open(path).convert("RGB")
        F_test = model.extract_features(img)
        score_map = compute_scoremap(F_test, F_mean)

        # ==== 4. 閾値処理と二値分類 ====
        THRESHOLD = 0.8 
        max_score = score_map.max()
        result = "Anomaly" if max_score > THRESHOLD else "Normal"
        filename = os.path.basename(path)
        print(f"- Result for {filename}: {result} (Max Score: {max_score:.4f})")
        
        # ==== 5. 保存 ====
        filename_without_ext = os.path.splitext(filename)[0]
        out_path = f"outputs/scoremaps/{filename_without_ext}_heatmap.png"
        save_scoremap(score_map, out_path)
        print(f"  [INFO] Heatmap saved to: {out_path}")

if __name__ == "__main__":
    main()
