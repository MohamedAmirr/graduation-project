from django import forms


class StoryForm(forms.Form):
    description = forms.CharField(
        label="Story Description", max_length=255, widget=forms.Textarea
    )
    num_scenes = forms.IntegerField(label="Number of Scenes", min_value=1)
