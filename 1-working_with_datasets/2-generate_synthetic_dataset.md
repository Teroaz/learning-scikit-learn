# Generating Synthetic Datasets

- Besides integrated datasets, scikit-learn offers functions to generate datasets that follow certain distributions.
- These synthetic data are useful for testing models and algorithms under controlled conditions.

#### 🔄 **Generating Datasets for Classification**

- **`make_classification` function:** Allows generating a random n-class classification dataset.
    - **Default parameter `n_classes`:** 2 (modifiable to change the number of classes).
    - The generated data include features and target vectors.

  ```python
  X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
  ```

- **Visualization:**

<figure>
  <img src="data_generated_from_make_classification.png" alt="Example of data generated with make_classification" width="450px" height="auto">
  <figcaption>Example of data generated with make_classification</figcaption>
</figure>

#### 📈 **Key Parameters of `make_classification`**

- `n_features`: Total number of features.
- `n_informative`: Number of features that are actually useful for predicting the target variable.
- `n_redundant`: Adds noise and complexity by introducing features that are linear combinations of informative features
  but without adding useful information.
- `n_clusters_per_class`: Number of clusters per class.

<hr/>

#### 🔄 **Generating Datasets for Regression**

- **`make_regression` function:** Creates a dataset for regression.
    - Can generate data with a linear relationship between the features and the target.

  ```python
  X, y = datasets.make_regression(n_features=1, n_informative=1)
  ```

- **Visualization:**

<figure>
  <img src="data_generated_from_make_classification.png" alt="Example of data generated with make_regression" width="450px" height="auto">
  <figcaption>Example of data generated with make_regression</figcaption>
</figure>

#### 📊 **Key Parameters of `make_regression`**

- Similar to those of `make_classification`, but adapted for regression.
