import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy
import time
from itertools import cycle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use('TkAgg')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
images = np.load(r"D:\MN\gastro three stage\SaveFileForStage1\updated\Stage1X.npy")
y = np.load(r"D:\MN\gastro three stage\SaveFileForStage1\updated\Stage1Y.npy")

# Reshape the input data to have 4 dimensions
X_train, X_test, y_train, y_test = train_test_split(images, y, stratify=y, test_size=0.1, random_state=2)

# Load the pre-trained model
model = tf.keras.models.load_model(r"D:\MN\gastro three stage\SaveFileForStage1\updated\TXception.h5")
model.summary()
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('LastXception').output)
intermediate_layer_model.summary()
# Get the number of layers
num_layers = len(intermediate_layer_model.layers)

print("Number of layers in the CNN model:", num_layers)

# Perform feature extraction
feature_engg_data1 = intermediate_layer_model.predict(images)
feature_engg_data1 = pd.DataFrame(feature_engg_data1)

# Standardize the feature-engineered data
x1 = feature_engg_data1.loc[:, feature_engg_data1.columns].values
x1 = StandardScaler().fit_transform(x1)

# Encode the labels
y1 = tf.keras.utils.to_categorical(y, num_classes=3)

######################################## PCA ############################################

pca_features = PCA(n_components=2)
x1 = pca_features.fit_transform(feature_engg_data1)
# x1 = pca_features.fit_transform(x1)
X_train, X_test, y_train, y_test = train_test_split(x1, y1, stratify=y1, test_size=0.1, random_state=2)

#######################################    DELM with multiple hidden layer    ###########################################################################
#
import numpy as np
import scipy.linalg
import time

# Define ReLU activation function
def relu(x):
    return np.maximum(x, 0)


# Define Deep ELM class
class DeepELM:
    def __init__(self, input_size, hidden_sizes):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.input_weights = [np.random.normal(size=[input_size, hidden_sizes[0]])]
        self.biases = [np.random.normal(size=[hidden_sizes[0]])]
        self.output_weights = None

        for i in range(1, self.num_layers):
            self.input_weights.append(np.random.normal(size=[hidden_sizes[i - 1], hidden_sizes[i]]))
            self.biases.append(np.random.normal(size=[hidden_sizes[i]]))

    def hidden_nodes(self, X):
        H = X
        for i in range(self.num_layers):
            G = np.dot(H, self.input_weights[i]) + self.biases[i]
            H = relu(G)
        return H

    def fit(self, X, y, alpha=None):
        H_train = self.hidden_nodes(X)
        if alpha is None:
            self.output_weights = np.dot(scipy.linalg.pinv(H_train), y)
        else:
            self.output_weights = np.linalg.lstsq(H_train.T @ H_train + alpha * np.eye(np.sum(self.hidden_sizes)),
                                                  H_train.T @ y, rcond=None)[0]

    def predict(self, X):
        H = self.hidden_nodes(X)
        predictions = np.dot(H, self.output_weights)
        return predictions


# Define the sizes of hidden layers
hidden_sizes = [1200, 2000, 1700]

# Create a DeepELM model
model = DeepELM(input_size=X_train.shape[1], hidden_sizes=hidden_sizes)

# Fit the model
start_time = time.time()
model.fit(X_train, y_train, alpha=None)  # Use alpha=None for Deep ELM
end_time = time.time()

start_time_DELM = time.time()
predictions = model.predict(X_test)
end_time_DELM = time.time()

# Convert probabilities to class predictions
predicted_classes_DELM = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print("Number of layers in the CNN model:", num_layers)

# Calculate accuracy
accuracy_DELM = accuracy_score(true_classes, predicted_classes_DELM)
print("Accuracy Deep ELM:", accuracy_DELM)

# Calculate precision, recall, F1-score for Deep ELM
precision_DELM = precision_score(true_classes, predicted_classes_DELM, average='weighted')
recall_DELM = recall_score(true_classes, predicted_classes_DELM, average='weighted')
f1_DELM = f1_score(true_classes, predicted_classes_DELM, average='weighted')
print("Precision Deep ELM:", precision_DELM)
print("Recall Deep ELM:", recall_DELM)
print("F1-score Deep ELM:", f1_DELM)

# Print testing times
print("Testing Time for Deep ELM: {:.10f} seconds".format(end_time_DELM - start_time_DELM))

# Print classification Score
cs_DELM = classification_report(true_classes, predicted_classes_DELM)

print("Classification Report DELM:", cs_DELM)

# Confusion Matrix for DELM
conf_matrix_DELM = confusion_matrix(true_classes, predicted_classes_DELM)
print("Confusion Matrix DELM:\n", conf_matrix_DELM)


########## ROC-AUC  ###########
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt

# ROC Curve for each class
lw = 1
n_classes = y_test.shape[1]
fpr_DELM = dict()
tpr_DELM = dict()
roc_auc_DELM = dict()
for i in range(n_classes):
    fpr_DELM[i], tpr_DELM[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc_DELM[i] = auc(fpr_DELM[i], tpr_DELM[i])

# Compute micro-average ROC curve and ROC area
fpr_DELM["micro"], tpr_DELM["micro"], _ = roc_curve(y_test.ravel(), predictions.ravel())
roc_auc_DELM["micro"] = auc(fpr_DELM["micro"], tpr_DELM["micro"])

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr_DELM[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr_DELM[i], tpr_DELM[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr_DELM["macro"] = all_fpr
tpr_DELM["macro"] = mean_tpr
roc_auc_DELM["macro"] = auc(fpr_DELM["macro"], tpr_DELM["macro"])

# Plot ROC curve for ensemble
plt.figure(figsize=(10, 8))
colors = cycle(['green', 'red', 'blue', 'purple', 'yellow', 'brown', 'gray', 'pink', 'orange', 'lime', 'aqua', 'darkorange', 'cornflowerblue', 'cyan', 'magenta', 'olive', 'indigo', 'teal', 'lavender', 'maroon', 'turquoise', 'tan', 'salmon', 'gold', 'lightgreen', 'skyblue', 'violet'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_DELM[i], tpr_DELM[i], color=color, lw=lw,
             label='C-{0} (area = {1:0.4f})'
             ''.format(i, roc_auc_DELM[i]))

plt.plot(fpr_DELM["macro"], tpr_DELM["macro"],
         label='macro-average ROC curve \n(area = {0:0.4f})'
               ''.format(roc_auc_DELM["macro"]),
         color='navy', linestyle=':', linewidth=2.5)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for TXception-PCA-DELM')
plt.legend(loc="lower right")
plt.show()
plt.savefig("D:/MN/gastro three stage/SaveFileForStage1/results/3rd stage new/TXception-PCA-DELM_roc-auc.png", dpi=600)

# Precision-Recall curve for each class
from sklearn.metrics import precision_recall_curve, auc

precision_DELM = dict()
recall_DELM = dict()
pr_auc_DELM = dict()

for i in range(n_classes):
    precision_DELM[i], recall_DELM[i], _ = precision_recall_curve(y_test[:, i], predictions[:, i])
    pr_auc_DELM[i] = auc(recall_DELM[i], precision_DELM[i])

# Compute micro-average precision-recall curve and its area
precision_micro, recall_micro, _ = precision_recall_curve(y_test.ravel(), predictions.ravel())
pr_auc_micro = auc(recall_micro, precision_micro)

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
colors = cycle(['green', 'red', 'blue', 'purple', 'yellow', 'brown', 'gray', 'pink', 'orange', 'lime', 'aqua', 'darkorange', 'cornflowerblue', 'cyan', 'magenta', 'olive', 'indigo', 'teal', 'lavender', 'maroon', 'turquoise', 'tan', 'salmon', 'gold', 'lightgreen', 'skyblue', 'violet'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall_DELM[i], precision_DELM[i], color=color, lw=1,
             label='C-{0} (area = {1:0.4f})'
             ''.format(i, pr_auc_DELM[i]))

# Plot micro-average precision-recall curve
plt.plot(recall_micro, precision_micro, color='deeppink', linestyle=':', linewidth=2.5,
         label='micro-average \n(area = {0:0.4f})'.format(pr_auc_micro))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve for TXception-PCA-DELM')
plt.legend(loc="lower right")
plt.show()
plt.savefig("D:/MN/gastro three stage/SaveFileForStage1/results/3rd stage new/TXception-PCA-DELM_pr.png", dpi=600)



##### CM ################
import seaborn as sns
# Plot confusion matrix with heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_DELM, annot=True, fmt='d', cmap='Blues', linewidths=.5, annot_kws={"fontsize":10})
plt.title('TXception-PCA-DELM')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
plt.savefig("D:/MN/gastro three stage/SaveFileForStage1/results/3rd stage new/TXception-PCA-DELM_cm.png", dpi=600, bbox_inches='tight', pad_inches=0.5)
######################################################


