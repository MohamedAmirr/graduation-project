from django.db import models

from authentication.models import Account


class Story(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    users = models.ManyToManyField(Account, related_name="stories")

    def __str__(self):
        return self.title


class Scene(models.Model):
    story = models.ForeignKey(Story, related_name="scenes", on_delete=models.CASCADE)
    sentence = models.TextField()
    image = models.ImageField(upload_to="story_images/")

    def __str__(self):
        return f"Scene for {self.story.title} - {self.sentence[:50]}"
