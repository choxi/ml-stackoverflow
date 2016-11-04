import pandas
from sklearn.tree               import DecisionTreeClassifier
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score
from sklearn.neighbors          import KNeighborsClassifier

data            = pandas.read_csv("data/2015 Stack Overflow Developer Survey Responses.csv", header=1)
data            = data[pandas.notnull(data["Compensation"]) & (data["Compensation"] != 'Rather not say')]
target          = data["Compensation"]

################################################################################
# Feature Selection
#   Options:
#       Country
#       Age
#       Gender
#       Tabs or Spaces
#       Years IT / Programming Experience
#       Occupation
feature_names   = ["Country", "Age", "Years IT / Programming Experience", "Occupation"]
features        = pandas.DataFrame([])

for feature in feature_names:
    one_hot     = pandas.get_dummies(data[feature])
    features    = pandas.concat([features, one_hot], axis=1)

################################################################################
# Train/Test Split
features_train, features_test, target_train, target_test = train_test_split(features, target)

################################################################################
# Benchmark
predictions = pandas.Series("$20,000 - $40,000", index=range(len(target_test)))
accuracy    = accuracy_score(predictions, target_test)
print "Benchmark Accuracy Score: %f" % accuracy

################################################################################
# Decision Tree
model = DecisionTreeClassifier()
model.fit(features_train, target_train)
predictions = model.predict(features_test)

accuracy = accuracy_score(predictions, target_test)
print "DT Accuracy Score: %f" % accuracy

################################################################################
# Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features_train, target_train)
predictions = model.predict(features_test)

accuracy = accuracy_score(predictions, target_test)
print "KNN Accuracy Score: %f" % accuracy
