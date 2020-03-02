# Help Navigate Robots

This is a competition question from Kaggle.

## Datasets

There are 2 groups of data in the Data folder:

* Data in career-con-2019 folder is downloaded from the offical site. But there is something wrong with Kaggle's scoring system that the submission will get wrong score. (You may need to download datasets from Kaggle, cause GitHub doesn't allow files larger than 50M, so this dataset probably is not complete)

* Other 4 csv files are created from the offical datasets, simplily by spliting last 20 samples of training set as testing set, while others remain training set.

You can also use the following code to generate training and testing data directly in Kaggle.

```python
import pandas as pd
# read Kaggle datasets
X_train = pd.read_csv('/kaggle/input/career-con-2019/X_train.csv')
y_train = pd.read_csv('/kaggle/input/career-con-2019/y_train.csv')
# split X_train
samples = 20
time_series = 128
start_x = X_train.shape[0] - samples*time_series
X_train_new, X_test_new = X_train.iloc[:start_x], X_train.iloc[start_x:]
# split y_train
start_y = y_train.shape[0] - samples
y_train_new, y_test_new = y_train.iloc[:start_y], y_train.iloc[start_y:]
```

## Project Overview

1. Import only training datasets since kaggle's sorcing system does not function correctly
2. Split training data sets into two parts: training & testing sets
3. Feature engineering
    * Since each sample covers 10 sensor channels and 128 measurements per time series, we need to group these measurements into one sample
    * Measurements are extracted by grouping the series on functions: max, min, median, mean, std, absolute maximum and quantiles
4. Use Random Forest Classifier (from scikit-learn) to train the model
5. Check model accuracy
    * OOB score
    * 10-Fold cross validation
    * 20 samples testing data

