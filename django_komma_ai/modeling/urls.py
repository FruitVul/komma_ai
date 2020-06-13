from django.contrib import admin
from django.urls import path
from.views import api_komma_pred

urlpatterns = [
    path('api/predict/', api_komma_pred, name='api_komma_pred'),
]
