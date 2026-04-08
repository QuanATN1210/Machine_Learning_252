from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LoadData:
    def __init__(
        self,
        path,
        image_size=(128, 128),
        color_mode="grayscale",
        normalize=True,
        train_subdir="train",
        test_subdir="test",
        test_size=0.2,
        random_state=42,
    ):
        self.path = Path(path)
        self.image_size = image_size
        self.color_mode = color_mode
        self.normalize = normalize
        self.train_subdir = train_subdir
        self.test_subdir = test_subdir
        self.test_size = test_size
        self.random_state = random_state
        self.idx2label = {}

    def load_data(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.path}")

        train_dir = self.path / self.train_subdir
        test_dir = self.path / self.test_subdir

        if train_dir.is_dir() and test_dir.is_dir():
            class_names = self._get_class_names([train_dir, test_dir])
            label2idx = {label: idx for idx, label in enumerate(class_names)}
            self.idx2label = {idx: label for label, idx in label2idx.items()}

            X_train, y_train = self._load_split(train_dir, label2idx)
            X_test, y_test = self._load_split(test_dir, label2idx)
            return X_train, y_train, X_test, y_test

        class_names = self._get_class_names([self.path])
        label2idx = {label: idx for idx, label in enumerate(class_names)}
        self.idx2label = {idx: label for label, idx in label2idx.items()}

        X, y = self._load_split(self.path, label2idx)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return X_train, y_train, X_test, y_test

    def _get_class_names(self, roots):
        class_names = set()
        for root in roots:
            for item in sorted(root.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    class_names.add(item.name)
        if not class_names:
            raise ValueError(f"Cannot find class folders in: {self.path}")
        return sorted(class_names)

    def _load_split(self, split_dir, label2idx):
        images = []
        labels = []

        for class_name, class_idx in label2idx.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                    images.append(self._load_image(image_path))
                    labels.append(class_idx)

        if not images:
            raise ValueError(f"No images found in: {split_dir}")

        return np.asarray(images, dtype=np.float32), np.asarray(labels)

    def _load_image(self, image_path):
        convert_mode = "L" if self.color_mode == "grayscale" else "RGB"

        with Image.open(image_path) as image:
            image = image.convert(convert_mode)
            image = image.resize(self.image_size)
            image = np.asarray(image, dtype=np.float32)

        if self.normalize:
            image /= 255.0

        return image


class BaseModel:
    supports_feature_importance = False

    def __init__(self, param_grid=None, cv=5, scoring="accuracy", search_n_jobs=-1, **kwargs):
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.search_n_jobs = search_n_jobs
        self.kwargs = kwargs
        self.model = None

    def build(self):
        raise NotImplementedError

    def fit(self, X_train, y_train):
        estimator = self.build()

        if self.param_grid:
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.search_n_jobs,
                return_train_score=False,
            )
            grid_search.fit(X_train, y_train)

            print("Grid search results:")
            for mean_score, std_score, params in zip(
                grid_search.cv_results_["mean_test_score"],
                grid_search.cv_results_["std_test_score"],
                grid_search.cv_results_["params"],
            ):
                print(f"mean_test_score={mean_score:.4f} std={std_score:.4f} params={params}")

            print(f"Best params: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            self.model = grid_search.best_estimator_
        else:
            estimator.fit(X_train, y_train)
            self.model = estimator

        return self.model

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X_test)

    def get_feature_importances(self):
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return None


class DecisionTreeModel(BaseModel):
    supports_feature_importance = True

    def build(self):
        return DecisionTreeClassifier(**self.kwargs)


class SVMModel(BaseModel):
    def build(self):
        return SVC(**self.kwargs)


class RandomForestModel(BaseModel):
    supports_feature_importance = True

    def build(self):
        return RandomForestClassifier(**self.kwargs)


class XGBoostModel(BaseModel):
    supports_feature_importance = True

    def build(self):
        from xgboost import XGBClassifier

        return XGBClassifier(**self.kwargs)


class LogisticRegressionModel(BaseModel):
    def build(self):
        return LogisticRegression(**self.kwargs)


class KNNModel(BaseModel):
    def build(self):
        return KNeighborsClassifier(**self.kwargs)



class HOGFeatureExtractor:
    def __init__(
        self,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt

    def extract(self, X):
        features = []

        for image in X:
            if image.ndim == 3:
                image = np.mean(image, axis=-1)

            feature = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                transform_sqrt=self.transform_sqrt,
                feature_vector=True,
            )
            features.append(feature)

        return np.asarray(features, dtype=np.float32)


class PipeLine:
    def __init__(self, data_load, feature_extractor, model, n_components=None):
        self.data_load = data_load
        self.feature_extractor = feature_extractor
        self.model = model
        self.n_components = n_components

    def run(self):
        X_train, y_train, X_test, y_test = self.data_load.load_data()
        
        ############# Feature extraction #############
        X_train = self.feature_extractor.extract(X_train)
        X_test = self.feature_extractor.extract(X_test)
        
        ############# Standardization #############
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ############## PCA ########################
        if self.n_components is not None:
            pca = PCA(n_components=self.n_components)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
        else:
            X_train_pca = X_train_scaled
            X_test_pca = X_test_scaled
            
        ############## Train #########################
        start_train = time.time()
        self.model.fit(X_train_pca, y_train)
        train_time = time.time() - start_train
        
        ############### Inference ######################
        start_inference = time.time()
        y_pred = self.model.predict(X_test_pca)
        inference_time = time.time() - start_inference

        idx2label = self.data_load.idx2label
        print(f"Train time: {train_time:.4f}s")
        print(f"Inference time: {inference_time:.4f}s")
        print("Classification report (test):")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=list(range(len(idx2label))),
                target_names=[idx2label[i] for i in range(len(idx2label))],
                zero_division=0,
            )
        )

        if self.model.supports_feature_importance:
            importances = self.model.get_feature_importances()
            if importances is not None:
                self._plot_feature_importance(importances)

    def _plot_feature_importance(self, importances):
        feature_indices = np.arange(len(importances))

        plt.figure(figsize=(14, 6))
        plt.bar(feature_indices, importances)
        plt.xlabel("PCA Feature Index")
        plt.ylabel("Importance")
        plt.title("Feature Importance After PCA")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data_load = LoadData(
        path=r"C:/Users/ADMIN/Desktop/252/ML/BTL/classification_task",
        image_size=(128, 128),
        color_mode="grayscale",
        normalize=True,
    )
    feature_extractor = HOGFeatureExtractor(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
    )

    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    model = RandomForestModel(
        param_grid=rf_param_grid,
        random_state=42,
        n_jobs=-1,
    )
    runner = PipeLine(data_load, feature_extractor, model, n_components=128)
    runner.run()




import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# --- PHẦN TRÍCH XUẤT ĐẶC TRƯNG SIFT (BAG OF VISUAL WORDS) ---
class SIFTFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1024, n_init=3)
        self.visual_words = None

    def _get_descriptors(self, images):
        descriptors_list = []
        for img in images:
            # Chuyển sang 8-bit nếu cần
            img_8bit = (img * 255).astype('uint8') if img.max() <= 1.0 else img.astype('uint8')
            # Nếu ảnh có 3 kênh (RGB), chuyển sang Gray
            if len(img_8bit.shape) == 3:
                img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY)
                
            kp, des = self.sift.detectAndCompute(img_8bit, None)
            if des is not None:
                descriptors_list.append(des)
            else:
                # Nếu không tìm thấy keypoint, trả về vector 0 để không lệch mảng
                descriptors_list.append(np.zeros((1, 128)))
        return descriptors_list

    def fit(self, X, y=None):
        print(f"SIFT: Đang gom cụm {self.n_clusters} từ điển hình ảnh...")
        descriptors_list = self._get_descriptors(X)
        all_des = np.vstack([d for d in descriptors_list if d is not None])
        self.kmeans.fit(all_des)
        return self

    def transform(self, X):
        print("SIFT: Đang tạo vector histogram...")
        descriptors_list = self._get_descriptors(X)
        features = np.zeros((len(X), self.n_clusters))
        
        for i, des in enumerate(descriptors_list):
            if des is not None and len(des) > 0:
                words = self.kmeans.predict(des)
                for w in words:
                    features[i][w] += 1
        
        # Normalize để vector có độ dài đơn vị (L2 norm)
        return normalize(features, norm='l2')

# --- PHẦN MÔ HÌNH SVM ---
def get_svm_model():
    """
    Hàm khởi tạo SVM theo yêu cầu của An
    """
    return SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)

# --- PHẦN MÔ HÌNH DECISION TREE (ĐỂ LẤY FEATURE IMPORTANCE) ---
def get_decision_tree_model():
    """
    Dùng để chụp ảnh Feature Importance theo yêu cầu nhóm trưởng
    """
    return DecisionTreeClassifier(max_depth=12, random_state=42)

# --- CÁCH CHẠY VÀ LẤY FEATURE IMPORTANCE TRÊN COLAB ---
"""
# 1. Trích xuất đặc trưng (Ví dụ dùng HOG theo pipeline chung của nhóm)
# extractor = HOGFeatureExtractor() 
# X_train_features = extractor.fit_transform(X_train)

# 2. Train Decision Tree
# dt_model = get_decision_tree_model()
# dt_model.fit(X_train_features, y_train)

# 3. Vẽ Feature Importance
# import matplotlib.pyplot as plt
# importances = dt_model.feature_importances_
# plt.figure(figsize=(10, 4))
# plt.bar(range(len(importances)), importances)
# plt.title("Feature Importance từ Decision Tree (An)")
# plt.savefig("feature_importance_an.png") # Chụp ảnh này lại gửi nhóm
# plt.show()
"""