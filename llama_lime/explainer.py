import openai
import os

class Explainer:
    def __init__(self, model, language_model = "openai/gpt-4"):
        self.model = model
        self.host, self.language_model = language_model.split("/")
        if self.host == "openai":
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("openai model chosen, missing `OPENAI_API_KEY` in environment")

    def explain(self, X, y=None):
        # Generate predictions
        y_pred = self.model.predict(X)

        # Generate explanations
        explanations = self._generate_explanations(X, y, y_pred)

        return explanations

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
