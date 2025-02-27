import pickle
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\Balaji\OneDrive\Desktop\Project\dataset_phishing.xlsx', engine='openpyxl')
df.head()
df.info()
df.columns
df['status'].value_counts()
mapping = {'legitimate': 0, 'phishing': 1}
df['status'] = df['status'].map(mapping)
df['status'].value_counts()
corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix['status']
threshold = 0.1
relevant_features = target_corr[abs(target_corr) > threshold].index.tolist()
X = df[relevant_features]
X = X.drop('status', axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
rf_predict = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, rf_predict)
print("Accuracy:{}%".format(round(accuracy * 100), 1))

rf_Accuracy_Score = accuracy_score(y_test, rf_predict)
rf_JaccardIndex = jaccard_score(y_test, rf_predict)
rf_F1_Score = f1_score(y_test, rf_predict)
rf_Log_Loss = log_loss(y_test, rf_predict)
rf_Precision = precision_score(y_test, rf_predict)
rf_Recall = recall_score(y_test, rf_predict)
print(f"Accuracy: {rf_Accuracy_Score}")
print(f"Jaccard Index: {rf_JaccardIndex}")
print(f"F1 Score: {rf_F1_Score}")
print(f"Log Loss: {rf_Log_Loss}")
print(f"Precision: {rf_Precision}")
print(f"Recall: {rf_Recall}")

rf_conf_matrix = confusion_matrix(y_test, rf_predict)
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

rf_report = classification_report(y_test, rf_predict)
print(rf_report)

svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_predict = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, svm_predict)
print("Accuracy:{}%".format(round(accuracy * 100), 1))

svm_Accuracy_Score = accuracy_score(y_test, svm_predict)
svm_JaccardIndex = jaccard_score(y_test, svm_predict)
svm_F1_Score = f1_score(y_test, svm_predict)
svm_Log_Loss = log_loss(y_test, svm_predict)
svm_Precision = precision_score(y_test, svm_predict)
svm_Recall = recall_score(y_test, svm_predict)
print(f"Accuracy: {svm_Accuracy_Score}")
print(f"Jaccard Index: {svm_JaccardIndex}")
print(f"F1 Score: {svm_F1_Score}")
print(f"Log Loss: {svm_Log_Loss}")
print(f"Precision: {svm_Precision}")
print(f"Recall: {svm_Recall}")

svm_conf_matrix = confusion_matrix(y_test, svm_predict)
sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

svm_report = classification_report(y_test, svm_predict)
print(svm_report)

params = { 'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'gamma': 0, 'objective': 'binary:logistic' }
xgb = XGBClassifier(**params)
xgb.fit(X_train_scaled, y_train)
xgb_predict = xgb.predict(X_test_scaled)
accuracy = accuracy_score(y_test, xgb_predict)
print("Accuracy:{}%".format(round(accuracy * 100), 1))

xgb_Accuracy_Score = accuracy_score(y_test, xgb_predict)
xgb_JaccardIndex = jaccard_score(y_test, xgb_predict)
xgb_F1_Score = f1_score(y_test, xgb_predict)
xgb_Log_Loss = log_loss(y_test, xgb_predict)
xgb_Precision = precision_score(y_test, xgb_predict)
xgb_Recall = recall_score(y_test, xgb_predict)
print(f"Accuracy: {xgb_Accuracy_Score}")
print(f"Jaccard Index: {xgb_JaccardIndex}")
print(f"F1 Score: {xgb_F1_Score}")
print(f"Log Loss: {xgb_Log_Loss}")
print(f"Precision: {xgb_Precision}")
print(f"Recall: {xgb_Recall}")

xgb_conf_matrix = confusion_matrix(y_test, xgb_predict)
sns.heatmap(xgb_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

xgb_report = classification_report(y_test, xgb_predict)
print(xgb_report)

tree = DecisionTreeClassifier()
tree.fit(X_train_scaled, y_train)
tree_predict = tree.predict(X_test_scaled)
accuracy = accuracy_score(y_test, tree_predict)
print("Accuracy:{}%".format(round(accuracy * 100), 1))

tree_Accuracy_Score = accuracy_score(y_test, tree_predict)
tree_JaccardIndex = jaccard_score(y_test, tree_predict)
tree_F1_Score = f1_score(y_test, tree_predict)
tree_Log_Loss = log_loss(y_test, tree_predict)
tree_Precision = precision_score(y_test, tree_predict)
tree_Recall = recall_score(y_test, tree_predict)
print(f"Accuracy: {tree_Accuracy_Score}")
print(f"Jaccard Index: {tree_JaccardIndex}")
print(f"F1 Score: {tree_F1_Score}")
print(f"Log Loss: {tree_Log_Loss}")
print(f"Precision: {tree_Precision}")
print(f"Recall: {tree_Recall}")

tree_conf_matrix = confusion_matrix(y_test, tree_predict)
sns.heatmap(tree_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

tree_report = classification_report(y_test, tree_predict)
print(tree_report)

nn = MLPClassifier()
nn.fit(X_train_scaled, y_train)
nn_predict = nn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, nn_predict)
print("Accuracy:{}%".format(round(accuracy * 100), 1))

nn_Accuracy_Score = accuracy_score(y_test, nn_predict)
nn_JaccardIndex = jaccard_score(y_test, nn_predict)
nn_F1_Score = f1_score(y_test, nn_predict)
nn_Log_Loss = log_loss(y_test, nn_predict)
nn_Precision = precision_score(y_test, nn_predict)
nn_Recall = recall_score(y_test, nn_predict)
print(f"Accuracy: {nn_Accuracy_Score}")
print(f"Jaccard Index: {nn_JaccardIndex}")
print(f"F1 Score: {nn_F1_Score}")
print(f"Log Loss: {nn_Log_Loss}")
print(f"Precision: {nn_Precision}")
print(f"Recall: {nn_Recall}")

nn_conf_matrix = confusion_matrix(y_test, nn_predict)
sns.heatmap(nn_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

nn_report = classification_report(y_test, nn_predict)
print(nn_report)

model = keras.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

dnn_loss, dnn_accuracy = model.evaluate(X_test_scaled, y_test)
print("Accuracy:{}%".format(round(dnn_accuracy * 100), 1))

dnn_predict = model.predict(X_test_scaled)
dnn_predict = (dnn_predict > 0.5).astype(int).ravel()
accuracy = accuracy_score(y_test, dnn_predict)
print("Accuracy:{}%".format(round(accuracy * 100), 1))

dnn_Accuracy_Score = accuracy_score(y_test, dnn_predict)
dnn_JaccardIndex = jaccard_score(y_test, dnn_predict)
dnn_F1_Score = f1_score(y_test, dnn_predict)
dnn_Log_Loss = log_loss(y_test, dnn_predict)
dnn_Precision = precision_score(y_test, dnn_predict)
dnn_Recall = recall_score(y_test, dnn_predict)
print(f"Accuracy: {dnn_Accuracy_Score}")
print(f"Jaccard Index: {dnn_JaccardIndex}")
print(f"F1 Score: {dnn_F1_Score}")
print(f"Log Loss: {dnn_Log_Loss}")
print(f"Precision: {dnn_Precision}")
print(f"Recall: {dnn_Recall}")

dnn_conf_matrix = confusion_matrix(y_test, dnn_predict)
sns.heatmap(dnn_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

dnn_report = classification_report(y_test, dnn_predict)
print(dnn_report)

pickle.dump(rf, open("rf_model.pkl", "wb"))
pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(xgb, open("xgb_model.pkl", "wb"))
pickle.dump(tree, open("tree_model.pkl", "wb"))
pickle.dump(nn, open("nn_model.pkl", "wb"))
model.save("dnn_model.h5")
pickle.dump(scaler, open("scaler.pkl", "wb")) 