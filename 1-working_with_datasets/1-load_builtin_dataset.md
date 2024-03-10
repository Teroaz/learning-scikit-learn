# Loading an Integrated Dataset

- Scikit-learn is an essential library for Machine Learning projects, offering a wide range of integrated datasets.
- Datasets play a crucial role as a starting point for machine learning projects.

#### 📚 **Integrated Datasets**

- **Examples of datasets for classification:** Iris, MNIST.
- **Example of dataset for regression:** Boston Housing Prices.
- These datasets are pre-loaded in scikit-learn and ready to use.

#### 💻 **Loading a Dataset**

- To access the datasets, import the `datasets` module from scikit-learn.

  ```python
  import sklearn.datasets as datasets
  ```

- **Example of loading:** The Iris dataset can be loaded with `load_iris()`.

  ```python
  iris = datasets.load_iris()
  ```

#### 🔍 **Structure of a Dataset**

- **Dictionary-like object:** Contains the data and metadata.
    - `data`: Contains the data (n_samples * n_features).
    - `target`: The data labels.
- **Key Attributes:**
    - Data size (`iris.data.shape`).
    - Target size (`iris.target.shape`).
    - Feature names (`iris.feature_names`).
    - Target label names (`iris.target_names`).

#### 🖥️ **Example Code and Important Attributes**

```
# Importing the library
import sklearn.datasets as datasets

# Loading the Iris dataset
iris = datasets.load_iris()

# Displaying key information
print("Description of the Iris dataset: {}".format(iris.DESCR))
print("Data size: {}".format(iris.data.shape))
print("Target size: {}".format(iris.target.shape))
print("The dataset has {} features, named {}".format(
    iris.data.shape[1], iris.feature_names))
print("The dataset contains {} samples, with the target label names {}".format(
    iris.data.shape[0], iris.target_names))
```

#### 📈 **Importance of Understanding Datasets**

- Knowing the structure and attributes of integrated datasets allows for better manipulation of these data for
  training models.
- It also helps understand how to prepare and use one's own datasets in a similar context.
