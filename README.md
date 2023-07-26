# LLama LIME

This Python library use the power of large language models to provide intuitive, human-readable explanations for the predictions made by machine learning models.

Features
Support for scikit-learn models
Integration with OpenAI's language models for explanation generation
Works with both classification and regression models
Installation
sh
Copy code
pip install ai_explainability
Quick Start
python
Copy code

from transformers import AutoModelForCausalLM, AutoTokenizer

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from ai_explainability import Explainer


iris = load_iris()
X, y = iris.data, iris.target

random_forest = RandomForestClassifier()
random_forest.fit(X, y)

# Create an explainer
explainer = Explainer(random_forest, language_model="openai/gpt-4")

explanations = explainer.explain(X)
```

For more detailed usage, see our Jupyter notebooks in the `examples/` directory.

Contributing
We welcome contributions! See our contribution guide for more details.

License
This project is licensed under the terms of the MIT license.