from django import forms


class MyForm(forms.Form):
    story = forms.CharField(max_length=100)
