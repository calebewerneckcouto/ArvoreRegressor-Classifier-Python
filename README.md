# House Prices Classification using Decision Tree

## Overview
This project aims to classify house sales based on the `SaleCondition` variable using a Decision Tree model. The dataset contains various attributes related to house sales, and missing values are handled appropriately before training the model.

## Dataset
The dataset used is `house_prices.csv`. The target variable for classification is `SaleCondition`.

## Preprocessing Steps
1. **Handling Missing Values**:
   - Numerical columns with missing values are filled with the mean.
   - Categorical columns are encoded using `LabelEncoder`.
   
2. **Feature Encoding**:
   - Categorical variables are transformed into numerical values using `LabelEncoder`.
   - The original categorical columns are dropped after encoding.

## Model Training
- The dataset is split into training (70%) and testing (30%) sets.
- A `DecisionTreeClassifier` is trained with `max_depth=10`, `max_leaf_nodes=20`, and `criterion='gini'`.
- Accuracy is computed for both training and testing sets.

## Model Evaluation
- The model's performance is evaluated using accuracy scores and a confusion matrix.
- Hyperparameter tuning is performed by testing different values of `max_depth` and `max_leaf_nodes`.
- A visualization of the decision tree is generated.

## Results
- The model achieved an accuracy of **85.6%** on the training set.
- The accuracy on the test set varied depending on hyperparameter tuning.
- A confusion matrix heatmap is generated for better interpretation of classification performance.

## How to Run
1. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn seaborn
   ```
2. Place `house_prices.csv` in the project directory.
3. Run the Python script to execute the preprocessing, model training, and evaluation steps.

## Visualization
- The decision tree is plotted using `plot_tree()` from `sklearn.tree`.
- The heatmap for the confusion matrix is generated using `seaborn`.

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

## Future Improvements
- Use more sophisticated feature engineering techniques.
- Try other classification models like Random Forest or Gradient Boosting.
- Perform more in-depth hyperparameter tuning to improve accuracy.

## Author
Calebe Werneck Couto

