from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function for visualization
def roc_auc(labels, scores, defect_name = None, save_path = None):
  fpr, tpr, _ = roc_curve(labels, scores, pos_label=1) # outlier label: 1
  roc_auc = auc(fpr, tpr)

  plt.title(f'ROC curve: {defect_name}')
  plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
  plt.legend(loc='lower right')
  plt.plot([0, 1], [0, 1], 'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')

  plt.show()
  return roc_auc

mat = scipy.io.loadmat('$path_for_data$.mat') 
mat = {k:v for k, v in mat.items() if k[0] != '_'}

print(mat.keys())
print("All dataset: {}".format(len(mat['y'])))

# Dictionary에서 normal, outlier에 해당하는 index를 찾아서 따로 저장
normal_data_index = [i for i in range(len(mat['y'])) if mat['y'][i] == 0]
outlier_data_index = [i for i in range(len(mat['y'])) if mat['y'][i] == 1]


print("normal data: {}".format(len(normal_data_index)))
print("outlier data: {}".format(len(outlier_data_index)))

# normal index, outlier index 출력
print(normal_data_index[:20])
print(outlier_data_index[:20])

# X (data), y (label)을 먼저 DataFrame으로 저장
dataframe_X = pd.DataFrame(mat['X'])
dataframe_y = pd.DataFrame(mat['y'])

# DataFrame에서 normal, outlier index의 행을 추출해서 분리
normal_data_X = dataframe_X.iloc[normal_data_index]
normal_data_y = dataframe_y.iloc[normal_data_index]

fraud_data_X = dataframe_X.iloc[outlier_data_index]
fraud_data_y = dataframe_y.iloc[outlier_data_index]

print("Length of dataframe_X: {}".format(len(dataframe_X)))
print("Length of dataframe_y: {}".format(len(dataframe_y)))

print("Length of normal_data_X: {}".format(len(normal_data_X)))
print("Length of normal_data_y: {}".format(len(normal_data_y)))

print("Length of abnormal_data_X: {}".format(len(fraud_data_X)))
print("Length of abnormal_data_X: {}".format(len(fraud_data_y)))

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(normal_data_X,normal_data_y,test_size=0.1,random_state=34)

print("Length of X_train: {}".format(len(X_train)))
print("Length of y_train: {}".format(len(y_train)))

print("Length of X_test: {}".format(len(X_test)))
print("Length of y_test: {}".format(len(y_test)))
     

# 테스트 시에는 Normal, Outlier 데이터를 모두 사용하므로 앞에서 만든 X_test, y_test에 outlier 데이터 추가
X_test = (X_test.append(fraud_data_X,sort=False)).sort_index()
y_test = (y_test.append(fraud_data_y,sort=False)).sort_index()

print("Length of X_test: {}".format(len(X_test)))
print("Length of y_test: {}".format(len(y_test)))


### GMM 선언 및 학습 ###
num_mixture = 100
GMM = GaussianMixture(n_components=num_mixture, random_state=42, covariance_type='diag')

GMM.fit(X_train)
y_test_score_gmm = -GMM.score_samples(X_test)

# Precision and recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_score_gmm, pos_label=1)

# Calculate average precision
average_precision = average_precision_score(y_test, y_test_score_gmm, pos_label=1, average = 'samples')

# For Visualization
plt.title("Precision-Recall Graph")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(recall, precision, "g", label = "Average Precision = %0.2F" % average_precision)
plt.legend(loc = "upper right")
plt.show()

print('average_precision:' , format(average_precision*100))


### KDE 선언 및 학습 ###
band_width = 0.05
KDE = KernelDensity(kernel='gaussian', bandwidth=band_width)

KDE.fit(X_train)
y_test_score_kde = -KDE.score_samples(X_test)

# Precision and recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_score_kde, pos_label=1)

# Calculate average precision
average_precision = average_precision_score(y_test, y_test_score_kde, pos_label=1, average = 'samples')

# For Visualization
plt.title("Precision-Recall Graph")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(recall, precision, "g", label = "Average Precision = %0.2F" % average_precision)
plt.legend(loc = "upper right")
plt.show()

print('average_precision:' , format(average_precision*100))


### LOF 선언 및 학습 ###
n_neighbors = 30
LOF = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto", novelty=True)

LOF.fit(X_train)
y_test_score_lof = -LOF.score_samples(X_test)

# Precision and recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_score_lof, pos_label=1)

# Calculate average precision
average_precision = average_precision_score(y_test, y_test_score_lof, pos_label=1, average = 'samples')

# For Visualization
plt.title("Precision-Recall Graph")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(recall, precision, "g", label = "Average Precision = %0.2F" % average_precision)
plt.legend(loc = "upper right")
plt.show()

print('average_precision:' , format(average_precision*100))


### iForest 선언 ###
max_samples = 200
IForest = IsolationForest(max_samples=max_samples, contamination = 'auto', random_state=0)

IForest.fit(X_train)
y_test_score_iforest = -IForest.score_samples(X_test)

# Precision and recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_score_iforest, pos_label=1)

# Calculate average precision
average_precision = average_precision_score(y_test, y_test_score_iforest, pos_label=1, average = 'samples')

# For Visualization
plt.title("Precision-Recall Graph")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(recall, precision, "g", label = "Average Precision = %0.2F" % average_precision)
plt.legend(loc = "upper right")
plt.show()

print('average_precision:' , format(average_precision*100))

X_test_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=5).fit_transform(X_test)

print(y_test_score_gmm.max())
print(y_test_score_gmm.min())
print(y_test_score_gmm.mean())
