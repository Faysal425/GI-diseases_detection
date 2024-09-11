################# 1st stage ###################
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib


matplotlib.use('TkAgg')

# Load the data
images = np.load("D:/MN/gastro three stage/SaveFileForStage1/updated/Stage1X.npy")
y = np.load("D:/MN/gastro three stage/SaveFileForStage1/updated/Stage1Y.npy")

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=2)

#model = tf.keras.models.load_model("D:/MN/Trash/CategoricalResult/BestLPCNNP.h5")
model = tf.keras.models.load_model(r"D:/MN/gastro three stage/SaveFileForStage1/updated/PSECNN.h5")


def SHAP_issue():
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough


SHAP_issue()
background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(X_test[[785,579, 129, 787, 410, 53]])

shap.image_plot(shap_values, X_test[[785,579, 129, 787, 410, 53]], show = True)
print(y_test[[785,579, 129, 787, 410, 53]])

#ORIGINAL IMAGE [785,579, 129, 787, 410, 53]
# TRUE CLASS = [1 0 2 1 0 2]


################# 2nd stage ###################
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')

# Load the data
images = np.load("D:/MN/gastro three stage/SaveFileForStage2/updated/Stage2X.npy")
y = np.load("D:/MN/gastro three stage/SaveFileForStage2/updated/Stage2Y.npy")

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=2)

model = tf.keras.models.load_model(r"D:/MN/gastro three stage/SaveFileForStage2/updated/PSECNN.h5")

def SHAP_issue():
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough


SHAP_issue()
background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(X_test[[785,579, 140, 787, 266, 778]])

shap.image_plot(shap_values, X_test[[785,579, 140, 787, 266, 778]], show = True)
print(y_test[[785,579, 140, 787, 266, 778]])


################# 3rd stage ###################
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')

# Load the data
images = np.load("D:/MN/gastro three stage/SaveFileForStage3/updated/Stage3X.npy")
y = np.load("D:/MN/gastro three stage/SaveFileForStage3/updated/Stage3Y.npy")

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.10, stratify=y, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train, random_state=2)

#model = tf.keras.models.load_model("D:/MN/Trash/CategoricalResult/BestLPCNNP.h5")
model = tf.keras.models.load_model(r"D:/MN/gastro three stage/SaveFileForStage3/updated/PSECNN.h5")


def SHAP_issue():
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough


SHAP_issue()
background = X_train[np.random.choice(X_train.shape[0], 20, replace=False)]
e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(X_test[[90, 365, 777, 300, 400, 77]])

shap.image_plot(shap_values, X_test[[90, 365, 777, 300, 400, 77]], show = True)
print(y_test[[90, 365, 777, 300, 400, 77]])
