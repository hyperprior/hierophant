import os

import openai
import torch


class Explainer:
    def __init__(self, model, language_model="openai/gpt-4"):
        self.model = model
        self.host, self.language_model = language_model.split("/")
        if self.host == "openai":
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError(
                    "openai model chosen, missing `OPENAI_API_KEY` in environment"
                )

    def explain(self, X, y=None, y_pred=None):
        if isinstance(self.model, torch.nn.Module):
            X_var = Variable(torch.FloatTensor(X))
            y_pred = self.model(X_var)
            y_pred = F.softmax(y_pred, dim=1).data.numpy()
        else:
            y_pred = self.model.predict_proba(X)

        return self._generate_explanations(X, y, y_pred)

    def _generate_explanations(self, X, y, y_pred):
        prompt = f"""
            The model architecture is {self.model}. The input data is {X} and the prediction is {y_pred}.
            Please explain why the predictions were what they were and explain it to a lay-person.
        """
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are creating the explanation for a machine learning model.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        return completion.choices[0].message.get("content")
