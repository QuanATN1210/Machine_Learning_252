from pathlib import Path
import time
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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


class MLPModel(BaseModel):
    """Multi-Layer Perceptron classifier built on sklearn's MLPClassifier.

    Default architecture: two hidden layers (512 → 256) with ReLU activation,
    Adam optimizer, and early stopping to avoid overfitting.
    All constructor kwargs are forwarded directly to MLPClassifier.
    """

    def build(self):
        defaults = dict(
            hidden_layer_sizes=(512, 256),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        # kwargs passed at construction override defaults
        defaults.update(self.kwargs)
        return MLPClassifier(**defaults)



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


class SIFTFeatureExtractor:
    def __init__(self, nfeatures=0, n_clusters=300, random_state=42, batch_size=2048):
        self.nfeatures = nfeatures
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size
        self.kmeans = None
        self._sift = None

    def _get_sift(self):
        if self._sift is not None:
            return self._sift

        try:
            import cv2
        except ImportError as error:
            raise ImportError(
                "OpenCV is required for SIFTFeatureExtractor. Install with `pip install opencv-contrib-python`."
            ) from error

        self._sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        return self._sift

    def _to_gray_uint8(self, image):
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0)
            arr = (arr * 255.0 if arr.max() <= 1.0 else arr).astype(np.uint8)
        return arr

    def _extract_descriptors(self, image):
        gray = self._to_gray_uint8(image)
        sift = self._get_sift()
        _, descriptors = sift.detectAndCompute(gray, None)
        return descriptors

    def fit(self, X):
        all_descriptors = []

        for image in X:
            descriptors = self._extract_descriptors(image)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)

        if not all_descriptors:
            raise ValueError("SIFT could not extract any descriptors from training images.")

        stacked = np.vstack(all_descriptors).astype(np.float32)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=self.batch_size,
            n_init=10,
        )
        self.kmeans.fit(stacked)
        return self

    def transform(self, X):
        if self.kmeans is None:
            raise ValueError("SIFTFeatureExtractor has not been fitted yet.")

        features = []
        for image in X:
            descriptors = self._extract_descriptors(image)

            hist = np.zeros(self.n_clusters, dtype=np.float32)
            if descriptors is not None and len(descriptors) > 0:
                visual_words = self.kmeans.predict(descriptors.astype(np.float32))
                hist = np.bincount(visual_words, minlength=self.n_clusters).astype(np.float32)

            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
            features.append(hist)

        return np.asarray(features, dtype=np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def extract(self, X):
        if self.kmeans is None:
            return self.fit_transform(X)
        return self.transform(X)


def prepare_dataset(data_load, feature_extractor=None):
    X_train, y_train, X_test, y_test = data_load.load_data()

    if feature_extractor is not None:
        if hasattr(feature_extractor, "fit_transform") and hasattr(feature_extractor, "transform"):
            X_train = feature_extractor.fit_transform(X_train)
            X_test = feature_extractor.transform(X_test)
        else:
            X_train = feature_extractor.extract(X_train)
            X_test = feature_extractor.extract(X_test)
    else:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrices after extraction, got X_train.ndim={X_train.ndim}, X_test.ndim={X_test.ndim}."
        )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Mismatched train samples: X_train has {X_train.shape[0]} rows but y_train has {y_train.shape[0]} labels."
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"Mismatched test samples: X_test has {X_test.shape[0]} rows but y_test has {y_test.shape[0]} labels."
        )

    return {
        "X_train": np.asarray(X_train, dtype=np.float32),
        "y_train": y_train,
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_test": y_test,
        "idx2label": data_load.idx2label,
    }


def save_prepared_dataset(prepared_data, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    idx2label_json = json.dumps({int(k): v for k, v in prepared_data["idx2label"].items()})
    np.savez_compressed(
        path,
        X_train=prepared_data["X_train"],
        y_train=prepared_data["y_train"],
        X_test=prepared_data["X_test"],
        y_test=prepared_data["y_test"],
        idx2label=np.asarray([idx2label_json], dtype=object),
    )


def load_prepared_dataset(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        idx2label = json.loads(str(data["idx2label"][0]))
        idx2label = {int(k): v for k, v in idx2label.items()}
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
            "idx2label": idx2label,
        }


class PipeLine:
    def __init__(
        self,
        data_load=None,
        feature_extractor=None,
        model=None,
        n_components=None,
        prepared_data=None,
    ):
        self.data_load = data_load
        self.feature_extractor = feature_extractor
        self.model = model
        self.n_components = n_components
        self.prepared_data = prepared_data
        self.train_times = []
        self.inference_times = []
        self.metrics_history = []

    def run(self):
        if self.model is None:
            raise ValueError("model is required to run the pipeline.")

        if self.prepared_data is None and self.data_load is None:
            raise ValueError("Either prepared_data or data_load must be provided.")

        if self.prepared_data is not None:
            X_train = self.prepared_data["X_train"]
            y_train = self.prepared_data["y_train"]
            X_test = self.prepared_data["X_test"]
            y_test = self.prepared_data["y_test"]
            idx2label = self.prepared_data.get("idx2label", {})
        else:
            X_train, y_train, X_test, y_test = self.data_load.load_data()
            idx2label = self.data_load.idx2label

            ############# Feature extraction #############
            if self.feature_extractor is not None:
                if hasattr(self.feature_extractor, "fit_transform") and hasattr(self.feature_extractor, "transform"):
                    X_train = self.feature_extractor.fit_transform(X_train)
                    X_test = self.feature_extractor.transform(X_test)
                else:
                    X_train = self.feature_extractor.extract(X_train)
                    X_test = self.feature_extractor.extract(X_test)
            else:
                # Learn directly on images: flatten (N, H, W) to (N, H*W)
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)
        
        ############# Standardization #############
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ############## PCA ########################
        if self.n_components is not None:
            pca = PCA(n_components=self.n_components, svd_solver="randomized", random_state=42)
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

        if not idx2label:
            idx2label = {idx: str(idx) for idx in np.unique(y_train)}

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        self.train_times.append(train_time)
        self.inference_times.append(inference_time)

        metrics = {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "precision": float(precision),
            "f1_score": float(f1_score),
            "train_time": float(train_time),
            "inference_time": float(inference_time),
            "train_times": list(self.train_times),
            "inference_times": list(self.inference_times),
        }
        self.metrics_history.append(metrics)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"F1-score (macro): {f1_score:.4f}")
        print(f"Train time: {train_time:.4f}s")
        print(f"Inference time: {inference_time:.4f}s")

        labels = sorted(idx2label.keys())
        target_names = [idx2label[i] for i in labels]

        print("Classification report (test):")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=labels,
                target_names=target_names,
                zero_division=0,
            )
        )

        if self.model.supports_feature_importance:
            importances = self.model.get_feature_importances()
            if importances is not None:
                self._plot_feature_importance(importances)

        return metrics

    def _plot_feature_importance(self, importances):
        feature_indices = np.arange(len(importances))

        plt.figure(figsize=(14, 6))
        plt.bar(feature_indices, importances)
        plt.xlabel("PCA Feature Index")
        plt.ylabel("Importance")
        plt.title("Feature Importance After PCA")
        plt.tight_layout()
        plt.show()
