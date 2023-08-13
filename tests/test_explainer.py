import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from llama_lime.explainer import Explainer


def test_explainer_with_sklearn_model():
    X, y = make_classification(n_samples=100, n_features=4)
    model = RandomForestClassifier()
    model.fit(X, y)

    explainer = Explainer(model, language_model="openai/gpt-4")
    explanation = explainer.explain(X[:1])

    assert isinstance(explanation, str)


def test_explainer_with_pytorch_model():
    class DummyPytorchModel(torch.nn.Module):
        def __init__(self):
            super(DummyPytorchModel, self).__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = self.linear(x)
            return x

    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    model = DummyPytorchModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    explainer = Explainer(model, language_model="openai/gpt-4")
    explanation = explainer.explain(X[:1].numpy())

    assert isinstance(explanation, str)
