# predictor/urls.py
from django.urls import path
from .views import home, predict_view, predict_api, analysis

urlpatterns = [
    path("", home, name="home"),
    path("predict/", predict_view, name="predict"),
    path("api/predict/", predict_api, name="predict_api"),
    path("analysis/", analysis, name="analysis"),
]