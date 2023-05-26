import graphviz
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris['data'][:, 2:]  # petal length and petal width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X, y)

# Generate the graphviz representation of the decision tree
dot_data = export_graphviz(
                tree_clf,
                out_file=None,
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True,
                special_characters=True
                )

graph = graphviz.Source(dot_data)
graph.view()





