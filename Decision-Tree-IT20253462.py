
import pandas as pd

df = pd.read_csv('dataset.csv')

df.head()
df.info()

# compile descriptive statistics for the data.
df.describe()

# Insert missing values
df.fillna('null', inplace=True)

# Categorical variables should be converted to numerical representations.
df["Father's occupation"] = df["Father's occupation"].astype("category")
df["Mother's occupation"] = df["Mother's occupation"].astype("category")
df["Debtor"] = df["Debtor"].astype("category")
df["Tuition fees up to date"] = df["Tuition fees up to date"].astype("category")
df["Scholarship holder"] = df["Scholarship holder"].astype("category")

# One-hot encoding or label encoding should be used to encode categorical variables.
df["Father's occupation"] = df["Father's occupation"].cat.codes
df["Mother's occupation"] = df["Mother's occupation"].cat.codes
df["Debtor"] = df["Debtor"].cat.codes
df["Tuition fees up to date"] = df["Tuition fees up to date"].cat.codes
df["Scholarship holder"] = df["Scholarship holder"].cat.codes

# Selecting relevant features
X = df[["Father's occupation", "Mother's occupation", "Debtor", "Tuition fees up to date", "Scholarship holder"]]
y = df['Target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Instantiate the classifier
clf = DecisionTreeClassifier()

# Model fitting using the training set of data
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Predicting labels for testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Calculate confusion matrix
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:", confusion)

# Calculate precision
precision = precision_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average=None)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred, average=None)

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average=None)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:", confusion)



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualize the decision tree
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True, ax=ax)
plt.show()


from sklearn.model_selection import GridSearchCV

# Set the hyperparameter grid parameters.
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Instantiate the decision tree classifier
clf = DecisionTreeClassifier()

# cross-validation grid search is carried out.
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train a decision tree with the best hyperparameter values
clf = DecisionTreeClassifier(**best_params)
clf.fit(X_train, y_train)

# Evaluate the performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
confusion = confusion_matrix(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:", confusion)



fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True, ax=ax)
plt.show()


fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True, ax=ax, fontsize=12)
plt.show()


fig, ax = plt.subplots(figsize=(20, 14))
plot_tree(clf, filled=True, ax=ax, fontsize=8)
plt.show()


fig, ax = plt.subplots(figsize=(36, 14))
plot_tree(clf, filled=True, ax=ax, fontsize=8)
plt.show()


fig, ax = plt.subplots(figsize=(50, 24))
plot_tree(clf, filled=True, ax=ax, fontsize=8)
plt.show()



from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Instantiate the decision tree classifier
clf = DecisionTreeClassifier()

# cross-validation grid search is carried out.
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# With the ideal hyperparameter values, train a decision tree.
clf = DecisionTreeClassifier(**best_params)
clf.fit(X_train, y_train)

# Evaluate the performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
confusion = confusion_matrix(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:", confusion)



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
# Visualize the decision tree
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True, ax=ax)
plt.show()


