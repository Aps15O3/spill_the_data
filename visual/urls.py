from django.contrib import admin
from django.urls import path,include
from visual import views

urlpatterns = [
    path('',views.home),
    path('home',views.home),
    path('linear',views.linear),
    path('linear2',views.linear3),
    path('linear3',views.linear2),
    path('logistic',views.logistic),
    path('logistic2',views.logistic2),
    path('intro',views.intro),
    path('logistic3',views.logistic3),
    path('svm',views.svm),
    path('svm1',views.svm1)

]
