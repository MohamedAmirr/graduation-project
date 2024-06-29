from django.conf.urls.static import static
from django.urls import path

from GP import settings
from .views import (
    home,
    generate_images,
    generate_story_view,
    display_story_view,
    check_user_email,
    user_stories_view,
    story_detail_view,
)

urlpatterns = [
    path("generate-story/", generate_story_view, name="generate_story"),
    path("generate-image/", generate_images, name="generate_images"),
    path("display-story/", display_story_view, name="display_story"),
    path("", home, name="home"),
    path("check_user_email/", check_user_email, name="check_user_email"),
    path("my-stories/", user_stories_view, name="user_stories"),
    path("story/<int:story_id>/", story_detail_view, name="story_detail"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
