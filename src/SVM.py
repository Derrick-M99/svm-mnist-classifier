import numpy as np
import matplotlib.pyplot as plt
import time
import cv2  
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D


mnist = fetch_openml('mnist_784', version=1, parser='auto')


X = mnist.data.astype(np.float32).to_numpy()  
y = mnist.target.astype(int)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.show()

X = X / 255.0  

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(X[0], bins=30, color='blue', alpha=0.7)
plt.title("Before MinMax Scaling")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

def apply_sobel_filter(images):
    processed_images = []
    for img in images:
        img_reshaped = img.reshape(28, 28).astype(np.uint8)  
        sobelx = cv2.Sobel(img_reshaped, cv2.CV_64F, 1, 0, ksize=3)  
        sobely = cv2.Sobel(img_reshaped, cv2.CV_64F, 0, 1, ksize=3)  
        sobel = np.sqrt(sobelx**2 + sobely**2)  
        processed_images.append(sobel.flatten())  
    return np.array(processed_images)

X_edges = apply_sobel_filter(X)  

def extract_hog_features(images):
    hog_features = []
    hog = cv2.HOGDescriptor((28,28), (14,14), (7,7), (7,7), 6)  
    for img in images:
        img_reshaped = (img.reshape(28, 28) * 255).astype(np.uint8)
        h = hog.compute(img_reshaped).flatten()
        hog_features.append(h)
    return np.array(hog_features)

X_hog = extract_hog_features(X)  

X_final = np.hstack((X, X_edges, X_hog))

scaler = MinMaxScaler()
X_final = scaler.fit_transform(X_final)

plt.subplot(1, 2, 2)
plt.hist(X_final[0], bins=30, color='red', alpha=0.7)
plt.title("After MinMax Scaling")
plt.xlabel("Pixel Intensity (Scaled)")
plt.ylabel("Frequency")

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.001, n_iters=1000, batch_size=256, tol=1e-4):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.tol = tol
        self.w = None
        self.b = None
        self.losses = []  
        self.accuracies = []  

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        prev_loss = float('inf')

        for epoch in range(self.n_iters):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            lr_t = self.lr / (1 + 0.001 * epoch)  

            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i:i + self.batch_size]
                batch_y = y_shuffled[i:i + self.batch_size]

                condition = batch_y * (batch_X @ self.w + self.b) >= 1
                d_w = 2 * self.lambda_param * self.w - np.sum((~condition)[:, None] * batch_y[:, None] * batch_X, axis=0)
                d_b = -np.sum((~condition) * batch_y)

                self.w -= lr_t * d_w / batch_X.shape[0]
                self.b -= lr_t * d_b / batch_X.shape[0]

            scores = X @ self.w + self.b
            hinge_loss = np.mean(np.maximum(0, 1 - y * scores)) + self.lambda_param * np.sum(self.w ** 2)
            self.losses.append(hinge_loss)

            predictions = np.sign(scores)
            accuracy = np.mean(predictions == y)
            self.accuracies.append(accuracy)

            if abs(prev_loss - hinge_loss) < self.tol:
                print(f"Early stopping at epoch {epoch}")
                break
            prev_loss = hinge_loss

    def predict(self, X):
        return X @ self.w + self.b  

class MultiClassSVM:
    def __init__(self):
        self.classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classifiers = {}

        for c in self.classes:
            print(f"Training SVM for digit {c} vs. all")
            y_binary = np.where(y == c, 1, -1)
            svm = SVM()
            svm.fit(X, y_binary)
            self.classifiers[c] = svm

    def predict(self, X):
        scores = np.array([svm.predict(X) for svm in self.classifiers.values()])
        return np.argmax(scores, axis=0)

start_time = time.time()
multi_svm = MultiClassSVM()
multi_svm.fit(X_train, y_train)
end_time = time.time()

print(f"Training Time: {end_time - start_time:.2f} seconds")
print(f"Training Accuracy: {accuracy_score(y_train, multi_svm.predict(X_train)) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, multi_svm.predict(X_test)) * 100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(multi_svm.classifiers[0].losses, label="Loss", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Hinge Loss")
plt.title("Loss vs Iterations")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(multi_svm.classifiers[0].accuracies, label="Training Accuracy", color='green')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Iterations")
plt.legend()

plt.show()

y_test_pred = multi_svm.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_final)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='tab10', alpha=0.5)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("PCA 3D Projection of MNIST Data")
plt.show()