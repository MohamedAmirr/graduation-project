from django.urls import path
from django.contrib.auth import views as auth_views
from .views import RegisterView, login_view, logout_view

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
]
