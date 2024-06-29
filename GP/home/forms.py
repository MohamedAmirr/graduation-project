import os

from django import forms
from openai import OpenAI


def check_words(sentence):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that checks for dummy, negative, and abusive words in a given text. "
                    "Dummy words are meaningless fillers like 'um', 'uh', 'like', etc. "
                    "Negative words are words with negative connotations such as 'bad', 'horrible', 'ugly', etc. "
                    "Abusive words are words that are offensive or insulting. "
                    "If the text contains any dummy, negative, or abusive words, respond with 'yes'. Otherwise, respond with 'no'."
                ),
            },
            {"role": "user", "content": sentence},
        ],
    )

    return chat_completion.choices[0].message.content


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
