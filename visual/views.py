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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

def linear(request):
    a=pd.read_csv('static/50Startups.csv')
    x=a[['R&D Spend','Administration','Marketing Spend']]
    y=a['Profit']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
    L=LinearRegression()
    L=L.fit(x_train,y_train)
    Yp=L.predict(x_test)
    fig = plt.figure(figsize=(9,11))
    sns.regplot(x=y_test,y=Yp,ci=None,color ='red')
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    context={
        'variable':r2_score(y_test, Yp) ,
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

def linear3(request):
    car = pd.read_csv("static/Cleaned Car.csv",index_col=[0])
    x=car.drop(columns="Price")#dependent variable
    y=car["Price"]
    ohe=OneHotEncoder()
    ohe.fit_transform(x[['name','company','fuel_type']])
    column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder="passthrough")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=433)##Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    r2_score(y_test,y_pred)
    fig = plt.figure(figsize=(9,11))
    plt.scatter(y_test,y_pred)
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test,y_test*m+b)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    if request.method=="POST":
        model = request.POST.get('model')
        year = request.POST.get('year')
        fuel = request.POST.get('fuel')
        company = request.POST.get('company')
        kms = request.POST.get('kms')

        car=pd.DataFrame([[model,company,year,kms,fuel]],columns=['name','company','year','kms_driven','fuel_type'])
        print(pipe.predict(car)[0])
       
        return HttpResponse(pipe.predict(car)[0])     
    else:
        print("no")
        context={
        "comp":sorted(x['company'].unique()),
        "models":sorted(x['name'].unique()),
        "f_type":sorted(x['fuel_type'].unique()),
        "yop":sorted(x['year'].unique()),
        "variable":r2_score(y_test,y_pred),
        "graph":data,
        "pred": "model"
        }   
        return render(request,"linear3.html",context)


def logistic(request):
    a=pd.read_csv('static/50Startups.csv')
    label=LabelEncoder()
    label=label.fit_transform(a['State'])
    temp=a
    temp['States']=label
    temp.drop(['State'],axis=1,inplace=True)
    X=temp.drop(['Profit'],axis=1)
    Y=temp['States']
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=2127)
    L=LogisticRegression()
    L.fit(x_train,y_train)
    L=L.predict(x_test)
    context={
        "variable":accuracy_score(y_test,L)*100
    }   
    return render(request,"logistic.html",context)

def logistic2(request):
    df=pd.read_csv('static/train.csv')
    df['LoanAmount'] =df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
    df.dropna(inplace=True)
    df['Loan_Status'].replace('Y',1,inplace=True)
    df['Loan_Status'].replace('N',0,inplace=True)
    df['Loan_Status'].value_counts()
    df.Gender=df.Gender.map({'Male':1,'Female':0})
    df.Married=df.Married.map({'Yes':1,'No':0})
    df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
    df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
    df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
    df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
    X = df.iloc[1:542,1:12].values
    y = df.iloc[1:542,12].values
    X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    lr_prediction = model.predict(X_test)

    context={
        "variable":accuracy_score(lr_prediction,y_test)
    }   
    return render(request,"logistic.html",context)

def home(request):
    transcript = request.POST.get('transcript') 
    if transcript=="show me an example of linear regression" or transcript=="Show me an example of linear regression.":
     
        return HttpResponse("linear")
    elif transcript=="show me another example of linear regression" or transcript=="linear regression example 2":
        return HttpResponse("linear2")
    elif transcript=="linear regression example 3":
        return HttpResponse("linear3")
    elif transcript=="tell me about machine learning":
        return HttpResponse("intro")        
    
    return render(request,'home.html')


def intro(request):
    return render(request,'intro.html')

def chart(request):
    car = pd.read_csv("static/Cleaned Car.csv",index_col=[0])
    x=car.drop(columns="Price")#dependent variable
    y=car["Price"]
    ohe=OneHotEncoder()
    ohe.fit_transform(x[['name','company','fuel_type']])
    column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder="passthrough")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=433)##Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    r2_score(y_test,y_pred)
    fig = plt.figure(figsize=(9,11))
    plt.scatter(y_test,y_pred)
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test,y_test*m+b)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    context={
        "graph":data,        
        }   
        
    return render(request,'chart.html',context)

# Create your views here.
