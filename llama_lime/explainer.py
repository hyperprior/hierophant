import os

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


import torch
import numpy as np

from functools import cached_property
from sklearn.inspection import permutation_importance
import shap
import os
from lime import lime_tabular



class Explainer:
    def __init__(
        self,
        model,
        features,
        feature_names,
        output=None,
        class_names=None,
        target_audience="data scientist trying to debug the model and better understand it",
        predictions=None,
        llm=ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")),
    ):
        self.model = model
        self.features = features
        self.feature_names = feature_names
        self.num_features = len(self.feature_names)
        self.output = output
        self.class_names = class_names
        self.predictions = predictions
        self.target_audience = target_audience
        self.llm = llm
        self.prompts = [
            SystemMessagePromptTemplate.from_template(
                "You are creating the explanation for a machine learning model whose architecture is {model}. Please explain why the model made the predictions it did knowing that your target audience is {target_audience}. Depending on your target audience, you should include numbers to support your findings. All numbers should be converted to scientific notation with 3 significant digits"
            )
        ]

    @cached_property
    def shap_values(self):
        # TODO keep check_additivity=False?
        explainer = shap.Explainer(self.model, self.features)
        return explainer(self.features, check_additivity=False).values

    def add_shap_values(self):
        prompt = """## SHAP
        
        SHAP (SHapley Additive exPlanations) values offer a measure of the contribution of each feature towards the prediction for a specific instance in contrast to a baseline value. They are based on Shapley values, a concept from cooperative game theory that assigns a payout (in this case, the prediction for an instance) to each player (in this case, each feature) based on their contribution to the total payout.

In more concrete terms, for a given instance, a SHAP value for a feature is calculated as the average difference in the model's output when that feature is included versus when it is excluded, considering all possible subsets of features. Positive SHAP values indicate that the presence of a feature increases the model's output, while negative SHAP values suggest that the presence of the feature decreases the model's output.

### Results
{self.shap_values}
"""

    @cached_property
    def feature_importances(self):
        importances = self.shap_values
        feature_importances = np.mean(np.abs(self.shap_values), axis=(0, 2))
        feature_importances = feature_importances / np.sum(feature_importances)

        if self.feature_names:
            feature_importances = dict(zip(self.feature_names, feature_importances))

        return feature_importances

    def add_feature_importances(self):
        base = """
## Feature Importance
            
Normalized feature importance is a way to measure the relative importance of each feature by taking into account the absolute contribution of each feature across all instances and classes. In the context of SHAP values, we first calculate the feature importance by finding the average absolute SHAP value for each feature across all instances and classes. We then normalize these importance values by dividing each one by the sum of all feature importances, ensuring that the total sums to 1. This gives us a measure of each feature's contribution relative to the total contribution of all features. This method assumes that the importance of a feature is proportional to the absolute magnitude of its SHAP values, irrespective of the direction (positive or negative) of its influence on the prediction.
            
### Results
        """
        prompt = base + " ".join(
            [
                f"Feature `{k}` has an importance of {v}."
                for k, v in self.feature_importances.items()
            ]
        )

        self.prompts.append(HumanMessagePromptTemplate.from_template(prompt))

    @cached_property
    def class_importances(self):
        class_importances = np.mean(np.abs(self.shap_values), axis=(0, 1))

        class_importances = class_importances / np.sum(class_importances)

        if self.class_names:
            class_importances = dict(zip(self.class_names, class_importances))

        return class_importances

    def add_class_importances(self):
        base = """## Class Importances

Class importance gives an indication of which classes are most often influenced by the features in a multi-class prediction problem. It is especially useful when you want to understand how each class is affected by the different features.

To calculate class importance, we use the SHAP values which measure the contribution of each feature to the prediction of each class for each instance. Specifically, we compute the average absolute SHAP value for each class across all instances and features. This is done by taking the absolute SHAP values (to consider the magnitude of influence regardless of direction), summing them across all instances and features for each class, and then dividing by the total number of instances and features. The result is a measure of the average influence of features on each class. The higher this value, the more a class is influenced by the features on average.

### Results

"""

        prompt = base + " ".join(
            [
                f"Class `{k}` has an importance of {v}."
                for k, v in self.class_importances.items()
            ]
        )

        self.prompts.append(HumanMessagePromptTemplate.from_template(prompt))

    @cached_property
    def instance_importances(self):
        instance_importances = np.sum(np.abs(self.shap_values), axis=(1, 2))
        return instance_importances / np.sum(instance_importances)

    def add_instance_importances(self):
        prompt = f"""## Instance Importances

Instance importance is a measure of how much each individual instance (or data point) is influenced by the features in your model. It is calculated by taking the sum of the absolute SHAP values for each instance across all features and classes.

This gives you an idea of how strongly the model's prediction is influenced by the features for each individual instance. Instances with higher importance values have predictions that are more strongly influenced by their features. This can be particularly useful if you want to understand which instances are driving your model's performance, or if you want to investigate instances where the model is particularly confident or uncertain.

### Results
{self.instance_importances}
"""
        self.prompts.append(HumanMessagePromptTemplate.from_template(prompt))

    @cached_property
    def feature_class_interactions(self, normalize=True):
        feature_class_importances = np.mean(np.abs(self.shap_values), axis=0)
        if normalize:
            feature_class_importances = feature_class_importances / np.sum(feature_class_importances, axis=0)
        feature_class_dict = {}
        for i, feature in enumerate(self.feature_names):
            feature_class_dict[feature] = {}
            for j, class_name in enumerate(self.class_names):
                feature_class_dict[feature][class_name] = feature_class_importances[i, j]

        return feature_class_dict
        
    def add_feature_class_interactions(self):
        prompt = """## Feature-Class Interaction

The feature-class interactions can be calculated by averaging the absolute SHAP values for each feature-class pair over all instances. This gives a measure of how much each feature contributes to the prediction of each class, on average. This can be useful for understanding how different features influence different classes, which can be particularly important in multi-class problems.

{self.feature_class_interactions}
"""



    @cached_property
    def permutation_importance(self):
        return permutation_importance(self.model, self.features, self.output)

    def add_permutation_importance(self):
        perm_dict = dict(
            zip(
                self.feature_names,
                list(
                    zip(
                        self.permutation_importance.importances_mean,
                        self.permutation_importance.importances_std,
                    )
                ),
            )
        )
        prompt = "## Permutation Importance Results:\n" + " ".join(
            [
                f"Feature `{k}` has a mean permutation importance of {v[0]} and a standard deviation of {v[1]}."
                for k, v in perm_dict.items()
            ]
        )

        self.prompts.append(HumanMessagePromptTemplate.from_template(prompt))

    @property
    def lime_explainer(self):
        return lime_tabular.LimeTabularExplainer(self.features, 
                                                            feature_names=self.feature_names, 
                                                            class_names=self.class_names, 
                                                            discretize_continuous=True)

    def lime_instances(self, X):
        all_explanations = []

        for instance in X:
            exp = self.lime_explainer.explain_instance(np.array(instance), self.model.predict_proba, num_features=self.num_features)
            all_explanations.append(exp.as_list())
        return all_explanations

    def add_lime_instances(self, X):
        prompt = f"""## LIME Instance Explainer
LIME, which stands for Local Interpretable Model-agnostic Explanations, is a method for explaining the predictions of any machine learning model. It was introduced in a paper by Ribeiro, Singh, and Guestrin in 2016.

Here's how LIME works:

Local surrogate model: LIME creates a local surrogate model around the specific instance you want to interpret. This model is simpler than the original model (often a linear model), and thus easier to interpret.

Perturbation: LIME perturbs the instance, creating a lot of slightly modified versions of it. It then uses the original model to predict the outcomes of these modified instances.

Weighting: LIME weights these new instances according to their proximity to the original instance. Those that are closer to the original instance get more weight.

Fit surrogate model: LIME fits the local surrogate model to the outcomes of the perturbed instances, taking into account the weights. This model is then used to interpret the prediction for the original instance.

By doing this, LIME can explain complex models locally (i.e., for specific instances) using simpler, interpretable models. The explanations provided by LIME are in the form of feature contributions, which tell you how much each feature contributed to the prediction for a specific instance. These contributions are represented as weights or coefficients in the local surrogate model.

It's important to note that LIME is model-agnostic, meaning it can be used with any type of machine learning model. It's also flexible and can be used for different types of data, including tabular data, text, and images.

### Output Format

The output from LIME is a list of tuples, where each tuple contains a feature and its corresponding weight in the explanation.

Each tuple corresponds to a feature and its impact on the prediction. The first element of the tuple is a statement about the feature's value, and the second element is the weight of that feature in the model's prediction.

{self.lime_instances(X)}
"""
        self.prompts.append(HumanMessagePromptTemplate.from_template(prompt))
        
    def explain(self, query: str ="Please give a detailed summary of your findings."):
        if isinstance(self.model, torch.nn.Module):
            X_var = Variable(torch.FloatTensor(self.features))
            y_pred = self.model(X_var)
            y_pred = F.softmax(self.predictions, dim=1).data.numpy()
        else:
            y_pred = self.model.predict_proba(self.features)

        return self._generate_explanations(
            self.features, self.output, self.predictions, query
        )


    def _generate_explanations(self, X, y, y_pred, query):
        self.prompts.append(
            HumanMessagePromptTemplate.from_template(
                "Based on this analysis please answer this question: {query}"
            )
        )
        chat_prompt = ChatPromptTemplate.from_messages(self.prompts)

        chat_prompt.format_messages(
            model=self.model, target_audience=self.target_audience, query=query
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=chat_prompt,
        )

        print(
            chain.run(
                {
                    "target_audience": self.target_audience,
                    "query": query,
                    "model": "model",
                }
            )
        )
