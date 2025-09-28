# predictor/forms.py
from django import forms

class IrisForm(forms.Form):
    sepal_length = forms.FloatField(min_value=0, label="Sepal length (cm)")
    sepal_width  = forms.FloatField(min_value=0, label="Sepal width (cm)")
    petal_length = forms.FloatField(min_value=0, label="Petal length (cm)")
    petal_width  = forms.FloatField(min_value=0, label="Petal width (cm)")