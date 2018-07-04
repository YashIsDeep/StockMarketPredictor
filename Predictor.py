from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import DataSource as ds
import matplotlib.pyplot as plt
import time

def getScore(df,p):
	X,y=ds.get_dataset(df,int(p))
	y=y.ravel()
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=int(round(time.time())))

	logisticRegr = LogisticRegression()
	logisticRegr.fit(x_train, y_train)
	score = logisticRegr.score(x_test, y_test)
	print(p,score)
	return float(score)

symbol='AAPL'# Apple
source='robinhood'
df=ds.getDataFrame(symbol , source)

p = range(30,90,1)
yp=[]
for each_p in p:
	score=getScore(df,each_p)*100
	yp=yp+[score]
t=np.array(p)
s=np.array(yp)
print(s[:10])
plt.plot(t ,s ,lw=2)
plt.ylim(0,100)
plt.show()