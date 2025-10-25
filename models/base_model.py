from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    """抽象モデルクラス。すべての特徴抽出モデルがこれを継承"""
    
    @abstractmethod
    def extract_features(self, img):
        """入力画像から特徴マップを返す"""
        pass
