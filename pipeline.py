from pathlib import Path
import time

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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
    def build(self):
        raise NotImplementedError


class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self):
        return SVC(**self.kwargs)


class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self):
        return RandomForestClassifier(**self.kwargs)


class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self):
        from xgboost import XGBClassifier

        return XGBClassifier(**self.kwargs)


class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self):
        return LogisticRegression(**self.kwargs)



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
        self.model = model.build()
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
    model = RandomForestModel(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    runner = PipeLine(data_load, feature_extractor, model, n_components=128)
    runner.run()
