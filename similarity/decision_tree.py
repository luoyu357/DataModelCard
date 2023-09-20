from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create a simple dataset and split it
data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
labels = ['A', 'A', 'B', 'B', 'C']
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


print(X_train, y_train)
print(X_test, y_test)
# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 1. Print Predicted Labels
y_pred = clf.predict(X_test)
print("Predicted Labels: ", y_pred)

# 2. Print Feature Importances
print("Feature Importances: ", clf.feature_importances_)

# 3. Print Decision Path
print("Decision Path: ")
print(clf.decision_path(X_test))

# 4. Print Tree Visualization
plot_tree(clf)
plt.show()
