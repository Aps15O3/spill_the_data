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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def linear(request):
    a=pd.read_csv('static/50Startups.csv')
    x=a[['R&D Spend','Administration','Marketing Spend']]
    y=a['Profit']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
    L=LinearRegression()
    L=L.fit(x_train,y_train)
    Yp=L.predict(x_test)
    fig = plt.figure(figsize=(5,5))
    sns.regplot(x=y_test,y=Yp,ci=None,color ='red')
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    if request.method=="POST":
         rd = request.POST.get('rd')
         market = request.POST.get('market')
         admin = request.POST.get('admin')        
         prof=pd.DataFrame([[rd,admin,market]],columns=['R&D Spend','Administration','Marketing Spend'])
         return HttpResponse(L.predict(prof))
    context={
        'variable':r2_score(y_test, Yp)*100 ,
        'graph':data,
    }
   
    return render(request,'linear.html',context)

def linear2(request):
    BostonHousing = pd.read_csv("static/BostonHousing.csv")
    Y = BostonHousing.medv
    X = BostonHousing.drop(['medv'], axis=1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1,random_state=2357)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    np.array(Y_test)
    fig = plt.figure(figsize=(6,6))#size x size y
    plt.subplot(2,1,1)
    plt.scatter(Y_test, Y_pred)
    m, b = np.polyfit(Y_test, Y_pred, 1)
    plt.plot(Y_test,Y_test*m+b)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    context={
        'variable':r2_score(Y_test, Y_pred)*100,
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
    fig = plt.figure(figsize=(6,5))
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
        "variable":r2_score(y_test,y_pred)*100,
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
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2202)
    L=LogisticRegression()
    L.fit(x_train,y_train)
    L=L.predict(x_test)
    
    
    context={
        "variable":accuracy_score(y_test,L)*100
    }   
    return render(request,"logistic.html",context)

def logistic2(request):
    df=pd.read_csv('static/Cleanedloan.csv',index_col=[1])
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=5712)
    
    model = LogisticRegression()
    model.fit(X_train,y_train)
    lr_prediction = model.predict(X_test)

    print(pd.DataFrame(X_train))
    context={
        "variable": accuracy_score(lr_prediction,y_test)
    }   
    if request.method=="POST":
        gender = int(request.POST.get('gender'))
        married = int(request.POST.get('married'))
        dependent = int(request.POST.get('dependent'))
        graduate = request.POST.get("graduate")
        selfemp = request.POST.get("selfemp")
        income = request.POST.get("income")
        co_applicant = float(request.POST.get("co_applicant"))
        loanamount = request.POST.get('loanamount')
        loanterm = request.POST.get("loanterm")
        credit = float(request.POST.get("credit"))
        property = request.POST.get("property")
        print(pd.DataFrame([[gender,married,dependent,graduate,selfemp,income,co_applicant,loanamount,loanterm,credit,property]],columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_term','Credit_History','Property_Area']))
       
        a=model.predict(pd.DataFrame([[gender,married,dependent,graduate,selfemp,income,co_applicant,loanamount,loanterm,credit,property]],columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_term','Credit_History','Property_Area']))
        print(a[0])
        return HttpResponse(a[0])
    return render(request,"logistic2.html",context)

def home(request):
    if request.method=="POST":
        transcript = request.POST.get('transcript') 
        transcript=transcript.lower()
        print(transcript)
    
        if transcript=="show me an example of linear regression" or transcript=="I want to see an example of linear regression":
         
            return HttpResponse("linear")
        elif transcript=="show me another example of linear regression" or transcript=="show me linear regression example 2":
            return HttpResponse("linear2")
        elif transcript=="show me linear regression example 3":
            return HttpResponse("linear3")
        elif transcript=="tell me about machine learning":
            return HttpResponse("intro") 
        elif transcript=="home" or transcript=="take me back to homepage" or transcript=="take me to homepage"  or transcript=="take me to home page" or transcript=="take me back to home page":
            return HttpResponse("home")        
        elif transcript=="show me an example of logistic regression":
            return HttpResponse("logistic") 
        elif transcript=="show me logistic regression example 2" or transcript=="show me logistic regression example to":
            return HttpResponse("logistic2")
        elif transcript=="show me logistic regression example 3":
            return HttpResponse("logistic3")
    
    return render(request,'home.html')


def intro(request):
    return render(request,'intro.html')



def logistic3(request):
    df = pd.read_csv('static/HDprediction_data.csv')
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    array = df.values
    X = array[:,0:df.shape[1]-1] #in
    Y = array[:,df.shape[1]-1] #out

# split between train and Test Sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=7,shuffle=True)

# Logistic regression model
    LR = LogisticRegression(solver='liblinear', random_state=0)
    LR.fit(X_train, Y_train) 
    context={
        "variable":LR.score(X_test, Y_test)*100
    }   
    return render(request,"logistic3.html",context)

# Create your views here.

def svm(request):
    dataset = pd.read_csv('static/PurchaseBehavior_data.csv')
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    classifier_kernel = SVC(kernel='rbf', random_state=0)
    classifier_kernel.fit(X_train,y_train)
    y_pred = classifier_kernel.predict(X_test)
    context={
        "score":accuracy_score(y_test, y_pred)*100
    }
    if request.method=="POST":
        age=request.POST.get("age")
        salary=request.POST.get("salary")
        pred=classifier_kernel.predict(pd.DataFrame([[age,salary]],columns=['Age','EstimatedSalary']))[0]
        return HttpResponse(pred)
    return render(request,"svm.html",context)
