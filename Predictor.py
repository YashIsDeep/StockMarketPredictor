from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import DataSource as ds
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def predictStocks(df,p):
	X,y=ds.get_dataset(df,int(p))
	y=y.ravel()
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=int(round(time.time())))

	Classifier = LogisticRegression()
	Classifier.fit(x_train, y_train)
	LRscore = Classifier.score(x_test, y_test)

	Classifier = RandomForestClassifier()
	Classifier.fit(x_train, y_train)
	RFScore = Classifier.score(x_test, y_test)

	score=(LRscore+RFScore)/2;
	if score>0.5:
		print(f"The stock prices will go up tomorrow (Confidence: {score})")
	else:
		confidence=1-score;
		print(f"The stock prices will go down tomorrow (Confidence: {confidence})")
def getScore(df,p):
	X,y=ds.get_dataset(df,int(p))
	y=y.ravel()
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=int(round(time.time())))

	Classifier = LogisticRegression()
	Classifier.fit(x_train, y_train)
	LRscore = Classifier.score(x_test, y_test)

	Classifier = RandomForestClassifier()
	Classifier.fit(x_train, y_train)
	RFScore = Classifier.score(x_test, y_test)
	print(p,LRscore,RFScore)
	return float(LRscore),float(RFScore)

symbol='AAPL'# Apple
source='robinhood'
df=ds.getDataFrame(symbol , source)

p = range(30,90,1)
y_lr=[]
y_rf=[]
for each_p in p:
	lrscore,rfscore=getScore(df,each_p)
	y_lr=y_lr+[lrscore]
	y_rf=y_rf+[rfscore]
t=np.array(p)
s1=np.array(y_lr)
s2=np.array(y_rf)
plt.plot(t ,s1 ,'r--' ,t ,s2 , 'gs')
plt.ylim(0,1)
plt.show()
print()
print('Enter optimal_p')
optimal_p=int(input())
predictStocks(df,optimal_p)