from django.contrib import admin
from django.urls import path,include
from visual import views

urlpatterns = [
    path('',views.home),
    path('home',views.home),
    path('linear',views.linear),
    path('linear2',views.linear2),
    path('linear3',views.linear3),
    path('logistic',views.logistic),
    path('logistic2',views.logistic2),
    path('intro',views.intro),
    path('logistic3',views.logistic3)

]
