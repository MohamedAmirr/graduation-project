import os

from django import forms
from openai import OpenAI

from home.helpers import check_words


class StoryForm(forms.Form):
    description = forms.CharField(
        label="Story Description", max_length=255, widget=forms.Textarea
    )
    num_scenes = forms.IntegerField(label="Number of Scenes", min_value=1)

    def clean_description(self):
        description = self.cleaned_data.get("description")

        if description:
            result = check_words(description)
            if result.lower() == "yes":
                raise forms.ValidationError(
                    "The description contains dummy, negative, or abusive words."
                )
        return description
