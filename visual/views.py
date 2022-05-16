from urllib import response
from django.http import HttpRequest,HttpResponse
from django.shortcuts import render,redirect
from multiprocessing import context
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

def linear(request):
    link = "https://data.covid19india.org/csv/latest/raw_data5.csv"
    if request.method=="POST":
        link = request.POST.get('link') 
    data=pd.read_csv(link)
    date=data['Date Announced'].str.split('/',expand=True)
    date.columns=['Day','Month','Year']
    data=pd.concat([data,date],axis=1)
    Day=data[data['Current Status']=='Hospitalized'].groupby(['Month','Day'])[['Num Cases']].sum()
    x=len(Day)
    x=np.arange(x)
    x=x.reshape(-1,1)
    y=Day.values
    from sklearn.linear_model import LinearRegression
    regressor=LinearRegression()
    regressor.fit(x,y)
    regressor.predict([[24]])
    Yp=regressor.predict(x)
    fig = plt.figure(figsize=(9,11))#size x size y
    plt.subplot(2,1,1)
    plt.scatter(x,y)
    plt.subplot(2,1,1)
    plt.plot(x,Yp)    
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    context={
        'variable':regressor.score(x,y)*100 ,
        'graph':data
    }
   
    return render(request,'linear.html',context)

def linear2(request):
    BostonHousing = pd.read_csv("static/BostonHousing.csv")
    Y = BostonHousing.medv
    X = BostonHousing.drop(['medv'], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    np.array(Y_test)
    fig = plt.figure(figsize=(9,11))#size x size y
    plt.subplot(2,1,1)
    plt.scatter(Y_test, Y_pred)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    context={
        'variable': model.coef_,
        'variable2': mean_squared_error(Y_test, Y_pred),
        'variable3':r2_score(Y_test, Y_pred),
        'graph':data
    }
    return render(request,'linear2.html',context)
    

def home(request):
    transcript = request.POST.get('transcript') 
    if transcript=="show me an example of linear regression" or transcript=="show me an example of linear regression.":
     
        return HttpResponse("linear")
        print(text)          
        if text == "hello":
            return render(request,'linear.html')
    elif transcript=="show me another example of linear regression" or transcript=="linear regression example 2":
        return HttpResponse("linear2")
    
    return render(request,'home.html')

# Create your views here.
