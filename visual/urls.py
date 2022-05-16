from django.contrib import admin
from django.urls import path,include
from visual import views

urlpatterns = [
    path('',views.home),
    path('home',views.home),
    path('linear',views.linear),
    path('linear2',views.linear2)
]
