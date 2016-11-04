import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data            = pandas.read_csv("data/2015 Stack Overflow Developer Survey Responses.csv", header=1)
data            = data[pandas.notnull(data["Compensation"]) & (data["Compensation"] != 'Rather not say')]
target          = data["Compensation"]
feature_names   = ["Country"]
features        = pandas.DataFrame([])

for feature in feature_names:
    one_hot     = pandas.get_dummies(data[feature])
    features    = pandas.concat([features, one_hot], axis=1)

features_train, features_test, target_train, target_test = train_test_split(features, target)

model = DecisionTreeClassifier()
model.fit(features_train, target_train)
predictions = model.predict(features_test)

accuracy = accuracy_score(predictions, target_test)

print "Accuracy Score: %f" % accuracy
