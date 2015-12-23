import csv
import numpy as np 
with open("adult-all.data", "rb") as csvfile:
	data_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

	csv.reader(csvfile)
	
	feature_names = ["age", "workingclass", "fnlwgt", 
	"education", "education-num", "marital-status", "occupation", 
	"relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
	#print feature_names

	#Load dataset, and target classes
	train_X, train_y = [], []
	for row in data_reader:
		if len(row) > 0:
			train_X.append(row[:-1])
			train_y.append(row[14]) 
	train_X = np.array(train_X)
	train_y = np.array(train_y)

print "number of instances: ", len(train_X)
#print "Features and target example: ", train_X[10], train_y[10]
#Lets transform the categorical classes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13]
#categorical_columns = [9]
for column in categorical_columns:
	enc = LabelEncoder()
	label_encoder = enc.fit(train_X[:, column])
	#print "Categorial classes: ", label_encoder.classes_
	integer_classes = label_encoder.transform(label_encoder.classes_).reshape(len(label_encoder.classes_), 1)
	enc = OneHotEncoder()
	ohe = enc.fit(integer_classes)
	num_of_rows = train_X.shape[0]
	t = label_encoder.transform(train_X[:, column]).reshape(num_of_rows, 1)
	new_features = ohe.transform(t)
	train_X = np.concatenate([train_X, new_features.toarray()], axis=1)
	
	for new_feat in label_encoder.classes_:
		feature_names.append(new_feat)
	del feature_names[column]

train_X = np.delete(train_X, [categorical_columns], 1)
enc = LabelEncoder()
label_encoder = enc.fit(train_y)
t = label_encoder.transform(train_y)
train_y = t

train_y = train_y.astype(float)
train_X = train_X.astype(float)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.25, random_state=33)
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion="entropy",
	max_depth=3, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)

from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True,
	show_classification_report=True, show_confussion_matrix=True):
	y_pred = clf.predict(X)
	if show_accuracy:
		print "Accuracy: {0:.3f}".format(
			metrics.accuracy_score(y, y_pred)), "\n" 

	if show_classification_report:
		print "Classification Report"
		print metrics.classification_report(y, y_pred), "\n"

	if show_confussion_matrix:
		print "Confusion Matrix"
		print metrics.confusion_matrix(y, y_pred), "\n"

measure_performance(X_train, y_train, clf, True, True, True)