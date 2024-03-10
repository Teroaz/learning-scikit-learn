# Data Preprocessing

#### 🌟 **Introduction to Data Preprocessing**

Data preprocessing is a crucial step in the machine learning process, involving the transformation of raw data into a
format more suitable for modeling. This stage is often the most time-consuming but is essential for achieving good
modeling results.

#### 🔄 **Preprocessing of Numerical Features**

##### **Scaling**

- **Objective:** Ensure that features are on the same scale so as not to bias models that rely on, for example, distance
  measurements.
    - **Common methods:**
        - **MinMax:** Transforms values to be between 0 and 1.
          ```python
          import sklearn.preprocessing as preprocessing
        
          minmax = preprocessing.MinMaxScaler() 
          # X is a matrix with float type
          minmax.fit(X)
        
          X_minmax = minmax.transform(X)
          ```

          **Result:**
          ```python
          # The original matrix
            [[  9    2 9288 3117]
            [   3    9 4850 5339]
            [   2    6  239 8628]
            [   5    8 9766 4451]]

          # The transform data using min-max scaler
            [[1.         0.         0.94982681 0.        ]
            [0.14285714 1.         0.48399286 0.40319361]
            [0.         0.57142857 0.         1.        ]
            [0.42857143 0.85714286 1.         0.24206133]]

        - **Standard:** Normalizes data to have a mean of 0 and a standard deviation of 1, assuming a normal
          distribution.
          ```python
          std = preprocessing.StandardScaler()
        
          # X is a matrix with float type
          std.fit(X)
        
          X_std = std.transform(X)
          ```

          **Result:**
          ```python
          # The original matrix
            [[  7    4 4955  896]
            [   6    5 6314 6703]
            [   5    5 5654 3841]
            [   5    7 5146 4904]]
          
          # The transform data using Standard scaler
            [[1.50755672 -1.14707867 -1.06855012 -1.51416544]
            [ 0.30151134 -0.22941573  1.51421486  1.24218525]
            [-0.90453403 -0.22941573  0.25989191 -0.1162917 ]
            [-0.90453403  1.60591014 -0.70555666  0.38827189]]
            ```

##### 🔄 **Non-linear Application on Numerical Features**

- **Binarizer:** Transforms values into 0 or 1 based on a defined threshold, useful for specific thresholds in the data.

  ```python
  binary = preprocessing.Binarizer(threshold=0.7)
  X_binary = binary.transform(Xb)
  ```

  **Result:**
  ```python
  # The original data
    [[0.2 0.4 0.9 0.7 0.1 0.8]
    [0.8 0.1 0.2 0.8 0.1 0.4]]
  
  # The transform data using Binarizer with threshold 0.7
    [[0. 0. 1. 0. 0. 1.]
    [1. 0. 0. 1. 0. 0.]]
  ```

#### 🔄 **Processing of Categorical Features**

##### **Label Encoder**

- **Objective:** Convert textual categorical labels into numerical values, allowing machine learning models to process
  them.

  ```python
  labelenc = preprocessing.LabelEncoder()
  labelenc.fit(targets)
  targets_trans = labelenc.transform(targets)
  ```

  **Result:**
    ```python
  # The original data
    ['Sun' 'Sun' 'Moon' 'Earth' 'Monn' 'Venus']

  # The transform data using LabelEncoder
    [3 3 2 0 1 4]
  ```

#### 💡 **Importance of Preprocessing**

- **"Garbage in, garbage out" principle:** The quality of input data directly affects the performance of models.
  Adequate preprocessing is essential to ensure the effectiveness of machine learning models.
- **Preprocessing = 70% of Time:** A large part of the work in machine learning projects is dedicated to cleaning and
  preparing data.
.