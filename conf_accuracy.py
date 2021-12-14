import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

'''
Gives the accuracy scores for various hog configurations. 

'''

config_dir = glob.glob("D:/config/*.p")
print(config_dir)
config_dir.remove('D:/config\\HSV_15_9_2.p')
final = []
for conf in config_dir:
	c = conf.split('\\')[1]
	print(c)
	dist_pickle = pickle.load( open(conf, "rb" ) )
	vec_fe = dist_pickle["vehicles"]
	nv_fec = dist_pickle["non-vehicles"]

	X = np.vstack((vec_fe, nv_fec)).astype(np.float64)        
               

	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)


	y = np.hstack((np.ones(len(vec_fe)), np.zeros(len(nv_fec))))

	# Split up data into randomized training and test sets
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)


	# Use a linear SVC 
	svc = LinearSVC()

	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()

	diff = t2-t
	accuracy = round(svc.score(X_test, y_test), 4)
	print(f"accuracy for : {conf} = {accuracy}")
	final.append([c,accuracy])
	print(final)

f = np.array(final)
df = pd.DataFrame(data=f, columns=["conf", "accuracy"])
print(df)
