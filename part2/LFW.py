from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import torch
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150
# Center data
mean = np.mean(X_train, axis=0)
X_train -= mean
X_test -= mean
#Eigen-decomposition
# U, S, V = np.linalg.svd(X_train, full_matrices=False)
# components = V[:n_components]
# eigenfaces = components.reshape((n_components, h, w))
# #project into PCA subspace
# X_transformed = np.dot(X_train, components.T)
# print(X_transformed.shape)
# X_test_transformed = np.dot(X_test, components.T)
# print(X_test_transformed.shape)

def pca_torch(X_train, X_test, n_components):
    """
    PCA implemented with PyTorch tensors, supporting CUDA/MPS/CPU.
    """
    # 自动选择设备：CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 转成张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32, device=device)

    # 中心化
    mean = X_train_t.mean(0, keepdim=True)
    X_train_c = X_train_t - mean
    X_test_c  = X_test_t - mean

    # 协方差矩阵
    cov = X_train_c.T @ X_train_c / (X_train_t.shape[0] - 1)

    # 特征分解
    eigvals, eigvecs = torch.linalg.eigh(cov)   # eigvecs: [n_features, n_features]

    # 取最大的 n_components
    idx = torch.argsort(eigvals, descending=True)[:n_components]
    components = eigvecs[:, idx].T              # [n_components, n_features]

    # 投影
    X_train_pca = X_train_c @ components.T
    X_test_pca  = X_test_c @ components.T

    # 转回 NumPy，方便后续 sklearn 用
    return (X_train_pca.cpu().numpy(),
            X_test_pca.cpu().numpy(),
            components.cpu().numpy(),
            eigvals.detach().cpu().numpy())   # 新增返回 eigvals



# PCA with PyTorch
X_transformed, X_test_transformed, components, eigvals = pca_torch(X_train, X_test, n_components)
eigenfaces = components.reshape((n_components, h, w))
print(X_transformed.shape)
print(X_test_transformed.shape)

import matplotlib.pyplot as plt
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()



# explained_variance = (S ** 2) / (n_samples - 1)
# total_var = explained_variance.sum()
# explained_variance_ratio = explained_variance / total_var
# ratio_cumsum = np.cumsum(explained_variance_ratio)
# print(ratio_cumsum.shape)
# eigenvalueCount = np.arange(n_components)
# plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
# plt.title('Compactness')
# plt.show()
explained_variance = eigvals[::-1]   # 注意 torch.linalg.eigh 返回的是升序，这里倒序
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var
ratio_cumsum = np.cumsum(explained_variance_ratio)

eigenvalueCount = np.arange(n_components)
plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
plt.title('Compactness')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#build random forest
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train) #expects X as [n_samples, n_features]
predictions = estimator.predict(X_test_transformed)
correct = predictions==y_test
total_test = len(X_test_transformed)
#print("Gnd Truth:", y_test)
print("Total Testing", total_test)
print("Predictions", predictions)
print("Which Correct:",correct)
print("Total Correct:",np.sum(correct))
print("Accuracy:",np.sum(correct)/total_test)
print(classification_report(y_test, predictions, target_names=target_names))