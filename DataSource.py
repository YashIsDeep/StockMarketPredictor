import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
def get_dataset(df,p):# Soft variables, can be modified as needed #Uses the prices of previous 'p' days as input and predicts whether the market go up or down on the next day
	
	shape=df.shape
	ndays=shape[0]
	nfeatures=shape[1]
	
	dataArray=df.values.ravel()
	X=np.array(dataArray[:nfeatures*p])
	
	UP=1
	DOWN=0
	iOK=df.keys().tolist().index('close_price') #Index of Key 'close_price'
	
	flag=flagger(dataArray[p*nfeatures+iOK],dataArray[(p-1)*nfeatures+iOK])
	y=np.array([flag])
	
	for i in range(1,ndays-p-1):
		t=np.array(dataArray[ (i*nfeatures):nfeatures*(p+i) ])
		X=np.vstack((X,t))
		flag=flagger(dataArray[(p+i)*nfeatures+iOK],dataArray[(p+i-1)*nfeatures+iOK])
		y=np.vstack((y,np.array([flag])))
	return X,y

def flagger(a,b):
	a=float(a)
	b=float(b)
	if b>a:
		return 1
	return 0

def getDataFrame(symbol,source):
	df=web.DataReader(symbol,source)
	#Remove details which are not required
	del df['interpolated']
	del df['session']
	return df