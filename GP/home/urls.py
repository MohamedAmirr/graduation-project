from django.urls import path

from .views import home, generate_images, generate_story_view, display_story_view

urlpatterns = [
    path("generate-story/", generate_story_view, name="generate_story"),
    path("generate-image/", generate_images, name="generate_images"),
    path("display-story/", display_story_view, name="display_story"),
    path("/", home, name="home"),
]
