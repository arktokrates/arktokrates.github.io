---
author: "Bernhard Fuchs"
layout: default
permalink: /KI/Machine-Learning/
last_modified_at: 2025-11-20
---


# Essentials for Machine Learning


Some notes on essential data structures and functions of Python for Machine Learning.



## Contents

1. [What is machine learning?](#what-is-machine-learning)

	1. [Types of machine learning](#types-of-machine-learning)

	1. [Machine learning pipeline](#machine-learning-pipeline)

	1. [Python packages for machine learning](#python-packages-for-machine-learning)


1. [Supervised machine learning](#supervised-machine-learning)

	1. [Preparing and shaping data](#preparing-and-shaping-data)

	1. [Overfitting and underfitting](#overfitting-and-underfitting)

	1. [Detecting and preventing overfitting and underfitting](#detecting-and-preventing-overfitting-and-underfitting)

	1. [Regularization](#regularization)


1. [Regression](#regression)

	1. [Regression types](#regression-types)

	1. [Linear regression](#linear-regression)

	1. [Working with linear regression](#working-with-linear-regression)

	1. [Critical assumptions for linear regression](#critical-assumptions-for-linear-regression)

	1. [Logistic Regression](#logistic-regression)

	1. [Data exploration using SMOTE](#data-exploration-using-smote)

	1. [Oversampling using SMOTE](#oversampling-using-smote)

	1. [Polynomial regression](#polynomial-regression)

	1. [Data preparation, model building, and performance evaluation (Polynomial regression)](#data-preparation-model-building-performance-evaluation-polynomial)

	1. [Ridge regression](#ridge-regression)

	1. [Data preparation, model building, and performance evaluation (Ridge regression)](#data-preparation-model-building-performance-evaluation-ridge)

	1. [Lasso Regression](#lasso-regression)

	1. [Data preparation, model building, and performance evaluation (Lasso regression)](#data-preparation-model-building-performance-evaluation-lasso)


1. [Classification](#classification)

	1. [What are classification algorithms?](#classfication-algorithms)

	1. [Types of classification](#types-of-classification)

	1. [Application types and selection of performance parameters](#application-types-and-selection-of-performance-parameters)

	1. [Naive Bayes](#naive-bayes)

	1. [Applying Naive Bayes classifier](#applying-naive-bayes-classifier)

	1. [Stochastic Gradient Descent (SGD)](#Stochastic-gradient-descent)

	1. [Applying Stochastic Gradient Descent](#applying-stochastic-gradient-descent)

	1. [K-nearest neighbour](#k-nearest-neighbour)

	1. [Applying K-nearest neighbours](#applying-k-nearest-neighbours)

	1. [Decision Tree](#decision-tree)

	1. [Applying decision tree](#applying-decision-tree)

	1. [Random forest](#random-forest)

	1. [Applying random forest regressor](#applying-random-forest)

	1. [Boruta](#Boruta)

	1. [Automatic feature selection with Boruta](#automatic-feature-selection-with-boruta)

	1. [Support vector machine](#support-vector-machine)

	1. [Applying support vector machine](#applying-support-vector-machine)

	1. [Cohen's kappa](#cohen's-kappa)






## What is machine learning? {#what-is-machine-learning}

- Machine learning coined by Arthur Samuel in 1959: The ability of a machine to learn from and replicate human behaviour.
- Main aim: Allow programs to learn automatically and make computers more intelligent (without any human intervention).
- Adoption accelerated in recent times because of these factors:
	1. Increased data capture through smart devices, phones, and IoT gadgets.
	2. Increased compuation power by edge devices (end-user devices)
	3. Better Internet connectivity and bandwiths

- Machine learning: ML refers to algorithms that learn and perform based on the data exposed to them.
- Deep learning: DL refers to layers of neural networks built with machine learning algorithms.
- Artificial intelligence: AI leverages different techniques, including ML and DL.

- Algorithm: a set of instructions used to solve problems
- ML algorithms are used to predict, classify, and improve the performance of any software application.

- Exemplary uses of ML: shape policies, make weather forecast, determine traffic rules etc.

- ML is dependent on data. The performance of an algorithm is evaluated on the quality of the input data.



### Types of machine learning {#types-of-machine-learning}

(criteria: algorithm can self-train and predict a condition vs. needs to identify patterns to derive outcomes)

- Supervised learning: data developed in supervised environments => inputs and outputs are known
	- Actions: Classification and regression => prediction and binary prediction
	- Linear regression
	- Logistic regression
	- Support vector machines
	- Decision trees
	- Examples: prediction of temperature increase by known year-wise temperature increase; prediction of crop yield; classification of waste

- Unsupervised learning: no supervisor to prepare data => outputs may not be known; find hidden patterns and recognize their relation
	- Examples: identification of anomalies over geographical landscapes; identification or user groups

- Reinforcment learning: the program learns from its previous errors, gets rewards for finding the correct solution (with reiteration)
	- Examples: YouTube recommendations for selecting songs or videos; game playing with bots; spell check; autocorrection



### Machine learning pipeline {#machine-learning-pipeline}

- A series of sequential steps used to codify and automate ML workflows to produce ML models
- End-to-end construct that orchestrates: data extraction, raw data input, preprocessing, features, outputs, model parameters, model training, deployment, predicting outputs
- ML pipelines are cyclical and iterative, until a successful algorithm is achieved (outcomes validated by a supervisor).

- MLOps (ML and Operations Professionals): a set of practices that combines ML, Devops, Data engineering
	=> ensure reliable and efficient deployment and maintenance of ML models in production systems
- Aims:
	- Improve communication and collaboration between MLOps
	- Shorten and manage complete development life cycle
	- Ensure contiuous delivery of high-quality predictive services

- Three phases are interconnected and influecne one another:
	- Design: understand the business and data, and then design the ML-powered software
	- Model development: verify the applicability of ML for the problem, by implementing proof of concept
	- Operations: deliver the developed ML model in production

- Some tools: Kubeflow, MLFlow, Data version control (DVC), Pachyderm, Metaflow, Kedro, Seldon Core, Flyte

- CI/CD pipeline automation:
	- Test and deploy new pipeline implementations automatically
	- Cope with dynamic changes in the data and business environment

- Automated Machine Learning (AutoML)
	- Combines the best practices in automation and machine learning
	- Enables organizations to build and deploy ML models using: predefined templates, frameworks, processes to speed up time to completion
	- Enhances the functionality of ML models
	- Some tools: Run: AI, Auto-Keras, H20AutoML, SMAC, AUTO-WEKA, AUTO-SKLEARN, AUTO-PYTORCH, ROBO



### Python packages for machine learning {#python-packages-for-machine-learning}

- Easily perform complex ML tasks and build ML models
- Rapidly build and test software prototypes

- Python packages: folders and modules that form the building blocks in Python-based programming
- Python libraries: a collection of packages or specific files containing prewritten code (imported into a code base)

- Some libraries for ML: NumPy, Pandas, TensorFlow, Aesara based on Theano, Matplotlib, SciPy, Scikit-learn, Keras, PyTorch

- NumPy, Pandas: manage preparation, loading, and manipulation of data
- TensorFlow, Aesara: used for fast numerical computing
- Matplotlib: used to plot data
- SciPy: solve mathematical equations and algorithms
- Scikit-learn: provides efficent versions of common algorithms to develop ML models
- Keras: makes implementation of neural networks easy
- PyTorch: spezializes in deep learning applications, and accelerates the path from prototyping to deployment


**Coding environment:**
- On local machines or cloud-based
- Google Colab: run Python notebooks on browser using the Google Cloud Platform (GCP) => for research and learning (not commercially)




## Supervised machine learning {#supervised-machine-learning}

- Machines are trained on labelled input => input data and correct output data are provided
- ML models identify patterns and methods, learn from them, and predict output
- An operator corrects incorrect predictions, until the algorithm achieves the highest accuracy.
- Algorithms:
	- Linear and logistic regression
	- Multi-class classification
	- Decision trees
	- Support vector machines

- Training data to eliminate false positives

- Two types:
	- Classification: segregate data into two or more categories
	- Regression: establish relationship between input and output variables; output variable as real or continuous value; example: predict the value of the stock market


- Applications: optimize and automate processes
	- HR: find right candidates for job vacancies
	- Finance: segregate good from bad loans
	- Quality inspection in manufacturing, defect level of damage
	- Forecasting for maritime industry (based on historic events and weather conditions)
	- Fraud protection models (e.g. Spam e-mails)
	- Waste management



### Preparing and shaping data {#preparing-and-shaping-data}

```python
import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv', sep=',')

df.head()
df.describe()
df.info()   # non-null values, data types

# Merging tables
df['Travelalone'] = np.where(df['SibSp'] + df['Parch'] > 0, 0, 1).astype('uint8')

# Drop unnecessary columns and save the dataframe to a new variable (no contribution to our model)
df1 = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1)   # axis = 1 => column-wise

# Check for missing values
df.isna().sum()   # Age, Cabin, Embarked

# Replace missing values
df['Age'].fillna(df1['Age'].median(skipna=True), inplace=True)

# Replace categorical values with a dummy variale (Pclass, Sex, Embarked) and creating a new data-variable
df_titanic = pd.get_dummies(df1, columns['Pclass', 'Sex', 'Embarked'], drop_first=True)   # Drop the first dummy values

# Preprocess the data and create a scalar to standardize the data points
X = df_titanic.drop(['Survived'], axis = 1)
y = df_titanic['Survived']

# Using MinMaxScaler or StandardScaler depends on the case
from sklearn.preprocessing import MinMaxScaler, StandardScaler

trans_MM = MinMaxScaler()
trans_SS = StandardScaler()

df_MM = trans_MM.fit_transform(X)   # Transform data between the range 0 and 1
pd.DataFrame(df_MM)    # Print the data

df_SS = trans_SS.fit_transform(X)   # Data have also negative values
pd.DataFrame(df_SS)    # Print the data
```



### Overfitting and underfitting {#overfitting-and-underfitting}

- Define how machine learning models are learning and applying what they learned.

- Bias: error introduced in the model
	- High bias: a big difference between the actual and the predicted values => not good for the model
	- Low bias: a low difference between the actual and the predicted values

- Variance: indicates how scattered data is
	- High variance: more scattered data
	- Low variance: less scattered data


**Overfitting:** a low bias and a high variance in the data

- Happens when a model focuses on too many details in the training dataset.
- Has a negative impact on the performance of the model on a new dataset.


**Underfitting:** a high bias and a high variance in the data

- Underfitting is easily detectable as it exhibits poor performance on the training dataset.

=> A model that performs well on training and testing data is a good model.

- If a model performs well with training data, but not with testing data, it is overfit.
- If a model does not perform well on both training data and testing data, it is underfit.




### Detecting and preventing overfitting and underfitting {#detecting-and-preventing-overfitting-and-underfitting}

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from mathplotlib import pyplot

# Create dataset and split features
X, y = make_classification(n_samples = 9000, n_features = 18, n_informative = 4, n_redundant = 12, random_state = 4)

# Create training and test datasets with a split of 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Create empty lists for train and test scores
train_scores, test_scores = list(), list()

# Create 21 values
values = [i for i in range(1,21)]

# Create a for loop for model and decision
for i in values:
	# Check how our model will be performing for different depths
	model = DecisionTreeClassifier(max_depth = i)
	model.fit(X_train, y_train)
	# Prediction for training and test data
	train_yhat = model.predict(X_train)
	train_acc = accuracy_score(y_train, train_yhat)
	test_yhat = model.predict(X_test)
	test_acc = accuracy_score(y_test, test_yhat)
	# Append the results to the empty score lists
	train_scores.append(train_acc)
	test_scores.append(test_acc)
	print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))   # Example: >20, train: 1.000, test: 0.937

# Plot the results for train and test score for different depth
pyplot.plot(values, train_scores, '-o', labels = 'Train')
pyplot.plot(values, test_scores, '-o', labels = 'Test')
pyplot.legend()
pyplot.show()


# Preventing overfitting by cross-validation
# Other methods: increase training set, remove unwanted features, regularization, boosting etc.
from sklearn.model_selection import GridSearchCV

# Create a grid
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 10, 20], 'min_samples_split': [5, 10, 20, 50, 100]}

# Create a GridSearch with a cross-validation of 3 and parallel processing (n_jobs = -1 => use all CPU cores available)
clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv= 3, n_jobs= -1, scoring = 'accuracy')

# Add the grid to the training dataset
clf.fit(X_train, y_train)

# Take the best parameters (or best estimator)
clf.best_estimator_   # DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=5)

# Print accuracy for both training and test dataset
print(accuracy_score(y_train, clf.best_estimator_.predict(X_train)))   # 0.970793
print(accuracy_score(y_test, clf.best_estimator_.predict(X_test)))   # 0.936666

# Reduced difference, needs more fine-tuning to make the difference as close as possible (resolve the overfitting problem)

# For solving underfitting problems, try out different models to increase the accuracy
```



### Regularization {#regularization}

- Regularization is a form of regression that shrinks the coefficient towards zero to reduce the complexity of the data.
- It reduces the variance of the model without an increase in bias, and prevents overfitting.
- Fitting involves a loss function: residual sum of squares (RSS) => the difference between the actual and the predicted value.


#### Types:

1. Dropout regularization
- Works by removing a random selection => the more units dropped out, the stronger is the regularization.
- Good for training neural networks

2. Early stopping
- Use a large number of epochs and plot the validation loss graph.
- When the validation loss moves from decreasing to increasing, stop training and save the model.

3. Co-adaptation
- Neurons predict patterns in the training data using output of specific neurons.
- If validation data does not have patterns that cause co-adaptation, it could cause overfitting.
- Dropout regularization reduces co-adoptation => it ensures that neurons cannot rely solely on other neurons.

4. Lasso-Regression (L1 regularization)
- It penalizes weights in proportion to the sum of the absolute values of the weights.
- It drives the weights of irrelevant or barely irrelevant features to exactly zero (removing those features from the model).

5. Ridge regression (L2 regularization)
- It shrinks coefficients close to zero for unimportant predictors (but never makes them exactly zero).
- The final model will include all the predictors.




## Regression {#regression}

- Regression is a supervised machine learning technique used to predict continuous values.
- It establishes a relationship between a independent variable x and a dependent variable y.
- It is the easiest and one of the most widely used machine learning algorithms.
- **Regression analysis** helps to understand how the value of a dependent variable changes corresponding to an independent variable.
- It predicts continuous or real values, e.g. temperature, age, salary, price, etc.
- **Regression algorithms** plot a best-fit line or a curve between the data.

- Sample application: oil and gas industry => linear and non-linear regression models are used to forecast global oil production

- Regression models are greatly used to predict events that have yet to occur.



### Regression types {#regression-types}

- Linear regression: popular regression technique
- Polynomial regression
- Support vector regression
- Decision tree regression
- Random forest regression
- Ridge regression
- Lasso regression
- Logistic regression


#### Linear regression
- Popular modeling technique
- Predicts a continuous dependent variable based on a independent variable.
- Uses least square criterion for estimation.
- Applied only if there is a linear relationship between the variables.


#### Polynomial regression
- A form of linear regression
- The relationship between an independent variable x and a dependent variable y is modeled as an nth degree polynomial.


#### Support vector regression
- Support vector machine regression (SVR) is a supervised learning algorithm.
- Aims to create the maximum data points between the boundary lines and the hyperplane.
- Used to solve both regression and classification problems.


#### Decision tree regression
- Commonly used supervised learning approach
- Builds a tree-like structure
- The internal nodes represent the “test” for an attribute.
- Branches represent the test results.
- The leaf nodes represent the final result or decision.
- It start from a root node (parent node) or dataset.
- The parent node splits into left and right child nodes or subsets of the dataset.
- The child nodes are further divided into children nodes (child nodes become the parent nodes of the children nodes).


#### Random forest regression
- An ensemble learing method that uses bagging or bootstrap aggregation techniques
- Combines multiple decision trees to predict the final output.
- Aggregated decision trees run in parallel and do not interact with each other.


#### Ridge regression
- Used when dealing with multicollinearity data
- The least squares are unbiased, and variances are large. => Predicted values will vary from the actual values.


#### Lasso regression
- LASSO: Least Absolute Shrinking and Selection Operator
- A form of linear regression
- Uses shrinkage to perform variable selection or feature selection.




### Linear regression {#linear-regression}

- Dependent variable is continuous, but independent variables can be continuous or discrete.
- The nature of the regression line is linear.
- The relationship between a dependent variable y and one or more dependent variables X are established using a regression line.


#### Simple linear regression
- Values of a numerical dependent variable are predicted using a single independent variable.
- Equation: y = mx + b   (m: coefficient of regression, b: intercept)


#### Multiple linear regression
- Uses more than one independent variable to predict the value of a numerical dependent variable.
- Equation: y = m_1x + m_2x + … + c
- Linear regression line: the line showing the relationship between dependent and independent variables
- Types of linear relationship:
	1. Positive linear relationship: The dependent variable increases on the y-axis, and the independent variable increases on the x-axis.
	2. Negative linear relationship: The dependent variable decreases on the y-axis, and the independent variable increases on the x-axis.

- Application of linear regression: e.g. the International Maritime Organization uses it to administer the sulphur cap rule.



### Working with linear regression {#working-with-linear-regression}

```python
import numpy as np
import pandas as pd
import mathplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

boston = datasets.load_boston()

df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['HousePrice'] = boston.target
df.head()

## Predict the price of a house based on the attributes available
df.describe()

# Checking for missing values
df.info()

# Plot a boxplot of the house prices
sns.boxplot(df['HousePrice'])   # There are outliers

# Deal with outliers: creating an upper and lower limit
upper_limit = df['HousePrice'].quantile(0.99)
lower_limit = df['HousePrice'].quantile(0.01)

# Replace the values below the lower limit and above the upper limit (capping)
df['HousePrice'] = np.where(df['HousePrice'] < lower_limit, lower_limit, df['HousePrice'])
df['HousePrice'] = np.where(df['HousePrice'] < upper_limit, upper_limit, df['HousePrice'])


# Check the linearity by importing statsmodel and fit OLS
import statsmodels.api as sm
X_constant = sm.add_constant(boston.data)
boston_model = sm.OLS(boston.target, boston.data).fit()
boston_model.summary()

def calculate_residuals(model, features, label):
	predictions = model.predict(features)
	df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
	df_results['Residuals'] = abs(df_results['Actual'] - df_results['Predicted'])
	return df_results

def linear_assumptions(model, features, label):
	df_results = calculate_residuals(model, features, label)

	sns.pyplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=7)   # height=7
	line_coords = np.arange(df_results.min().min(), df_results.max().max())
	plt.plot(line_coords, line_coords, color='darkorange', linestyle='--')
	plt.title('Actual vs. Predicted')
	plt.show()

linear_assumptions(boston_model, boston.data, boston.target)   # Showing linearity between Actual vs. Predicted (till x = 40)

# In some of the cases when you do not have linearity, you can either use linear transformation or you can square your turn.   (term?)


## Plot the correlation matrix
corr = df.corr()   # calculate the correlation
corr.style.background_gradient(cmap='coolwarm')   # Darker red fields are positively correlated, and darker blue fields are negatively correlated.

# Check for multicollinearity between the attributes => use VF (variance inflation factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
x = df.drop(['HousePrice'], axis = 1)   # Dropping the target variable
vif_data = pd.DataFrame()
vif_data['features'] = x.columns

# Calculate VIF
vif_data['vif'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]
print(vif_data)   # Showing the VIF value for independent variables

# => 5–10 is critical, above 5 is highly correlated

# Remove all the variables that are highly correlated
df1 = df.drop(['NOX', 'RM', 'AGE', 'PTRATIO'], axis = 1)


# Create independent and dependent variable for splitting the data
x = df1.drop(['HousePrice'], axis = 1)
y = df1['HousePrice']

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Consider a test size of 25 %
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.25)

# Fit the OLS model
model = sm.OLS(y_train, X_train).fit()
model.summary()   # Check for R-square and P-values

# Fit the regression model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)

# Print the results for training data
print("R-Square: {}".format(r2_score(y_train, y_pred_train)))
print("MAE: {}".format(mean_absolute_error(y_train, y_pred_train)))   # Mean absolute error
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))   # Root mean squared error

# Print the results for test data
print("R-Square: {}".format(r2_score(y_test, y_pred_test)))
print("MAE: {}".format(mean_absolute_error(y_test, y_pred_test)))
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))

# => Difference between RMSE for training and test data
```



### Critical assumptions for linear regression {#critical-assumptions-for-linear-regression}

- Regression is a parametric approach where one makes assumptions of data for analysis. => This makes regression restrictive.
- Therefore, successful regression analysis requires validation of the assumptions made.


**Important assumptions:**

1. The dependent or the response variables and the independent or predictor variables have a linear and additive relationship.
	- Linear relationship implies that a change in response variable Y due to one unit change in predictor variable X remains constant.
	- In additive relationship, the effect of X on the response Y is independent of other variables.
2. Error terms are normally distributed.
3. Independent variables are not correlated. => Multicollinearity between independent variables does not exist.
4. Error or residual terms are not correlated. => Autocorrelation is absent.
5. The error terms have constant variance. => They show homoscedasticity. (non-constant variance is called heteroscedasticity)




### Logistic Regression {#logistic-regression}

- Logistic regression is used to predict a data value based on prior observations of a dataset.
- Finds relationship between qualitative discrete dependent variables and several independent variables.
- It is a machine learning method used in classification problems to distinguish one class from another.
- The simplest case is a binary classification. => The algorithm answers a given question:
	- If the answer is positive, it belongs to a set of positive points.
	- If the answer is negative, it belongs to the negative point class.
- Many problems require a probaility estimate as output. => This algorithm is efficient in calculating probability.
- The main goal is to accurately predict the class of two possible label data points.
- A logistic regression model ensures that the output always falls between 0 and 1. => Sigmoid function: $f(x) = \frac{1}{1 + e^{-x}}$
- We use the sigmoid function to map prediction to probabilities.

**Types of logistic regression:**
- Binary (e.g. 0 or 1)
- Multinomial (e.g. 0, 1, or 2)

- Decision boundary: The probability score is between 0 and 1, based on the inputs provided.

- Sample applications:
	- Prediction of a cricket match outcome
	- Assessment of presence or absence of fraud regarding financial transactions (based on banking activity)
	- Prediction of bankruptcy



### Data exploration using SMOTE {#data-exploration-using-smote}

```python
import numpy as np
import pandas as pd
import mathplotlib.pyplot as plt

df = pd.read_csv('cuisines.csv')

df.head()   # Based on the ingredients, we have to predict which cuisine a dish belongs to.

df.info()

# Plot how many cuisines we have
df.cuisine.value_counts().plot.barh()

# Create a dataframe for each of the cuisines
thai_df = df[(df.cuisine == 'thai')]
japanese_df = df[(df.cuisine == 'japanese')]
chinese_df = df[(df.cuisine == 'chinese')]
indian_df = df[(df.cuisine == 'indian')]
korean_df = df[(df.cuisine == 'korean')]

print(f'Thai df: {thai_df.shape}')
print(f'Japanese df: {japanese_df.shape}')
print(f'Chinese df: {chinese_df.shape}')
print(f'Indian df: {indian_df.shape}')
print(f'Korean df: {korean_df.shape}')

# Create a function to check how many ingredients have been used for each cusine (EDA)
def create_ingredient_df(df):
	ingredient_df = df.T.drop(['cuisine', 'Unnamed: 0']).sum(axis = 1).to_frame('value')
	ingredient_df = ingredient_df[(ingredient_df.T != 0).any()] 
	ingredient_df = ingredient_df.sort_values(by='value', ascending=False, inplace=False)
	return ingredient_df

# Check ingredients of Thai cuisine and plot the first ten of them
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()

# Check ingredients of Japanese cuisine and plot the first ten of them
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()

# Check ingredients of Chinese cuisine and plot the first ten of them
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()

# Check ingredients of Indian cuisine and plot the first ten of them
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()

# Check ingredients of Korean cuisine and plot the first ten of them
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()


# Drop common ingredients used so they do influence our model
feature_df = df.drop(['cuisine', 'Unnamed: 0', 'rice', 'garlic', 'ginger'], axis = 1)
labels_df = df.cuisine
feature_df.head()

# Different numbers of ingredients between the cuisines => imbalanced dataset (influenced by the Korean cuisine)


## To balance this dataset, we use a technique called SMOTE
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
transformed_feature_df, transformed_labels_df = oversample.fit_resample(feature_df, labels_df) 

print(f'New label count: {transformed_labels_df.value_counts()}')   # All cuisines have now 799 ingredients
print(f'Old label count: {labels_df.value_counts()}')


# Fit the logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Split out data into training and test datasets with a ratio of 70:30
X_train, X_test, y_train, y_test = train_test_split(transformed_feature_df, transformed_labels_df, test_size=0.3)

# Fit the logistic regression
lr = LogisticRegression(multi_class='ovr', solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print('Accuracy is {}'.format(accuracy))   # 80 %

# Check the probability of each of the ingredients
print(f'Ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
print(f'Cuisine: {y_test.iloc[50]}')   # Korean

# Test what our model will predict
test = X_test.iloc[50].values.reshape(-1,1).T
proba = model.predict_proba(test)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)

# Sort the values
toppred = resultfd.T.sort_values(by=[0], ascending=False)
toppred.head()   # Probability of each of the cuisines

# Check the classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))   # How the model performs for different outcomes
```



### Oversampling using SMOTE {#oversampling-using-smote}

- Imbalanced dataset: a dataset that contains observations in a class that is higher or lower than other classes
- To correct data bias, perform oversampling by the synthetic minority using SMOTE.
- SMOTE: data augmentation technique => adds to the minority class examples to balance data distribution
- It generates training records between existing minority instances using linear interpolation.
- SMOTE algorithm:
	1. Select an input vector from the minority class.
	2. Find its k-nearest neighbours.
	3. Build a line joining the point under consideration and the chosen neighbour
	4. Place a synthetic point anywhere on the line drawn
	5. Repeat steps 1–4 until data is balanced

- Applications: SMOTE is used to balance data for classification problems. It helps to increase decision boundaries and classification performance.




### Polynomial regression {#polynomial-regression}

- Expresses the relationship between a dependent variable and an independent variable in a linear manner.
- When data is more complex, we cannot use linear models to fit non-linear data.
- Polynomial regression is a statistical method used in machine learning for predictive modeling and analysis.
- Models the non-linear relationship between the dependent variable (y) and independent variable (x) as an n-th degree polynomial.
- Equation: $Y = \theta_0 + \theta_{1x} + \theta_{2x^2}$
- Considered as a linear model due to linear nature of coefficients or weights associated with features.
- To capture more data points and convert original features into high-order, we use the polynomial features class provided by scikit-learn.

- Applications:
	- Develop climate model predictions
	- Predict the rise of different diseases within populations and their spread rate
	- Examine the generation of any synthesis



### Data preparation, model building, and performance evaluation (Polynomial regression) {#data-preparation-model-building-performance-evaluation-polynomial}

```python
# How to fit the polynomial regression

import numpy as np
import pandas as pd
import mathplotlib.pyplot as plt

dataset = pd.read_csv('position_salaries.csv')
dataset.info()

# Split the columns into features and target dataset
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Check for linear regression model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fitting a linear regression line based on salary and position level
def viz_linear():
	plt.scatter(X, y, color='red')
	plt.plot(X, lin_reg.predict(X), color='blue')
	plt.title('Linear regression model')
	plt.xlabel('Position level')
	plt.ylabel('Salary')
	plt.show()
	return

viz_linear()


# Fit polynomial regression
from sklearn.preprocessing import PolynomialFeatures

# Fit polynomial features at degree 4
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Function for polynomial regression
def viz_polynomial():
	plt.scatter(X, y, color='red')
	plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
	plt.title('Linear regression with Polynomial of degree 4')
	plt.xlabel('Position level')
	plt.ylabel('Salary')
	plt.show()
	return

viz_polynomial()   # The regression line is able to fit the majority of the data points


# Preditions for one value
lin_reg.predict([[5.5]])   # array([249500.])

pol_reg.predict(poly_reg.fit_transform([[5.5]]))   # array([132148.43750002])
```



### Ridge regression {#ridge-regression}

- Ridge regression shrinks the coefficient towards zero to reduce complexity of data.
- This technique discourages the use of complex models and reduces the risk of overfitting.
- It decreases the multicollinearity between features in a dataset.
- It minimizes the variance of the model, without increasing its bias.
- This regularization technique involves a loss function: residual sum of squares (RSS). => the difference between the actual and predicted values
- The constant or tuning parameter $\lambda$ decides the appropriate rate to penalize the flexibility.
- Penalty is a multiple of $\lambda$ and the sum of squares of weights. => When $\lambda$ equals to 0, the penalty term has no effect.
- When $\lambda \rightarrow \infty$ goes towards infinity, the impact of the shrinkage penalty grows. => The ridge regression coefficient estimates will approach zero.
- This results in a less complex dataset, which helps us to get the best fitting model.

- Applications: eliminate multicollinearity in data models
	- Hospitality industry: Manage seasonal fluctuations in booking prices of hotels or resorts
	- Farming: Predict grain yield under different water regimes



### Data preparation, model building, and performance evaluation (Ridge regression) {#data-preparation-model-building-performance-evaluation-ridge}

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('housing.csv')
df.head()   # MEDV is the dependent variable or target which we will be predicting.

df.info()

# Create independent and dependent variables
X = df.drop(['MEDV'], axis = 1)
y = df['MEDV']

# Split the data into a training and test dataset with ratio of 75:25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fit the model (keep alpha at 1)
Ridge_model = ridge(alpha=1).fit(X_train, y_train)
Ridge_model.intercept_   # 24.878370 (intercept value)

# Mean square error
y_pred = Ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))   # 4.741358

# Check the coefficients and R2-score
Ridge_model.coef_
r2_score(y_test, y_pred)   # 0.678975

## Use GridSearch
from sklearn.model_selection import GridSearchCV
cv = RepeatedKFold(n_splits=10, n_repeat=3, random_state=1)

# Define a grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.1)
model = Ridge()
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# Read the results
results = search.fit(X_train, y_train)
print('MAE: %.3f' % results.best_score_)   # MAE: -3.500
print('Config: %.3f' % results.best_params_)   # Config: {'alpha': 0.7000000000000001}

# Fit with alpha points and try to predict
# Check the evolution matrix
Ridge_model = Ridge(alpha=0.7).fit(X_train, y_train)
y_pred = Ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))   # 4.731437

r2_score(y_test, y_pred)   # 0.680317

# Check the coefficients of Ridge regression
pd.Series(Ridge_model.coef_, index = X_train.columns)   # A lot of negatives (penalization)
```



### Lasso Regression {#lasso-regression}

- Lasso regression is a form of regression that shrinks the coefficient towards zero, in order to reduce the complexity of the data.
- Cf. Ridge regression forces the coefficient very close to zero, but never makes it exactly zero.
- Lasso makes some of the coefficient estimates to be exactly equal to zero. It takes help of a large tuning parameter $\lambda$.
- Therefore, Lasso performs variable selection or feature selection.
- Lasso reduces the learning of more complex data and overfitting of the model.
- It decreases the variance of the model, without an increase in bias.
- The fitting procedure involves a loss function: residual sum of squares (RSS). => the difference between the actual and predicted values
- The variation differs from Ridge regression only in penalizing the high coefficients.
- The sum of the absolute value of weights is used instead of squaring the weights.

- Applications: applied in datasets to shrink parameter estimates towards zero => variable selection or feature selection
	- Healthcare sector: distinguish normal epithelial or stromal tissue from cancer tissue => determine the spread of prostate cancer in a patient
	- Insurers use it to determine whether or not to give insurance by analyzing the lifestyle of a customer through social media profiles.



### Data preparation, model building, and performance evaluation (Lasso regression) {#data-preparation-model-building-performance-evaluation-lasso}

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
import mathplotlib.pyplot as plt

df = pd.read_csv('Hitters.csv')
df.head()

df.info()   # 322 rows, 21 columns; some null values in 'Salary'

# Replace the null values in 'Salary' with the median
df['Salary'].fillna(df['Salary'].median(skipna=True), inplace=True)
df.isna().sum()   # The missing values have been replaced.

# Create a dummy variable for categorical aspects (converting)
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']], drop_first=True)

# Create X and y variables
y = df['Salary']
x_ = df.drop(['Unnamed: 0', 'Salary', 'League', 'Division', 'NewLeague'], axis = 1).astype('float64')
X = pd.concat([x_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)

# Split the data into a training and test dataset with ratio of 75:25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create the Lasso model
lasso_model = Lasso().fit(X_train, y_train)
lasso_model.intercept_   # 342.873393

lasso_model.coef_   # Last attribute 'NewLeague' has been penalized to zero.

# Try to optimize it. But before, check the attitudes like RMSE and R2-score
y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))   # 345.619069

r2_score(y_test, y_pred)   # 0.365751

# Optimizing using cross-validation
lasso_cv_model = LassoCV(alphas = np.random.randint(0, 1000, 100), cv=10, max_iter=10000, n_jobs=-1).fit(X_train, y_train)

lasso_cv_model.alpha_   # 14 (best alpha value got from cross-validation)

# Use this alpha value to check if we can tune our model
lasso_tuned = Lasso().set_params(alpha = 14).fit(X_train, y_train)
y_pred_tuned = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_tuned))   # 346.708392

# Check the coefficients of the tuned model
pd.Series(lasso_tuned.coef_, index=X_train.columns)
```



## Classification {#classification}


### What are classification algorithms? {#classfication-algorithms}

- Classification is an activity that classifies things into sub-categories. => done by an automated process or a machine
- It identifies the set of categories to which a new observation belongs.
- Classification is done on the basis of a training dataset containing prior observations and whose membership categories are known.
- Classification algorithms are used to classify or categorize data into a class or category. => Classifies that map input data into a specific category.
- Classifiers operate on both structured and unstructured data.
- Common classification algorithms:
	1. Naive Bayes
	2. Linear classifiers: logistic regression
	3. Decision tree classifier
	4. K-nearest neighbours
	5. Stochastic gradient descent (SGD) classifier
	6. Random forests

- To determine the classifier to be used:
	1. Read the nature of input data
	2. Determine the nature
	3. Decide on the classifier to be used

- Some applications:
	- Detection of health problems
	- Classification of the nature of certain diseases like cancer
	- Recognition of face and speech
	- Navigation of self-driving cars

- Businesses use machine learning algorithms to reduce bias in outcomes, and to improve quality of products.



### Types of classification {#types-of-classification}

- Binary classification
- Multi-class classification
- Multi-label classification
- Imbalanced classification


#### Binary classification

- Divides instances into two classes, based on a classification rule.
- Typically, one class is known as the traditional state (class label 0) and the other, abnormal state (class label 1).
- Example: a juicy and fresh apple vs. a rotten apple
- Some of the popular algorithms used for binary classification:
	- Logistic regression
	- K-nearest neighbours
	- Decision tree
	- Support vector machine
	- Naive Bayes



#### Multi-class classification

- Multi-class classification classifies instances into more than two classes.
- Classification tasks can have only one label assigned to them. Example: a banyan tree or a palm tree (not both)
- Examples: Plant classification, optical character recognition
- Multi-class classification does not have the notion of normal and abnormal outcomes.
- It classifies as belonging to one among all the range of known classes.
- Some of the popular algorithms used for multi-class classification:
	- K-nearest neighbours
	- Decision tree
	- Naive Bayes
	- Random forest
	- Gradient boost



#### Multi-label classification

- Multi-label classification refers to the tasks that have two or more class labels.
- It requires specialized algorithms (those for binary and multi-class classification are not suitable).
- The algorithms commonly used are:
	- Multi-label decision trees
	- Multi-label random forests
	- Multi-label gradient boosting



#### Imbalanced classification

- Imbalanced classification involves unequal distribution of classes.
- Most of the training data belongs to traditional class and very less to the abnormal class (cf. binary classification).
- Specialized techniques:
	- Random undersampling
	- SMOTE oversampling



### Application types and selection of performance parameters {#application-types-and-selection-of-performance-parameters}

- Decide on type of classification and apply classifiers on input data
- The resulting model output is available with a probability or a category.
- The effectiveness of the model is measured by certain metrics.
- Choose features while applying classifiers to the dataset under consideration.
- Use metrics to monitor and measure model performance on a validation set. => Get feedback on whether the approach is working or not.

- Look at the confusion matrix to evaluate the performance of a classifier.
- Confusion matrix is an intuitive and simple metric to assess the correctness and accuracy of a model.

![Confusion matrix](/assets/images/ML_Confusion_matrix.png)

	- True Positive (TP): positive case classified as positive
	- True Negative (TN): negative case classified as negative
	- False Positive (FP): negative case classified as positive
	- False Negative (FN): positive case classified as negative

- It is used for classification problems where the outputs are often of two or more sorts of classes.
- In an ideal scenario, the model correctly classifies and predicts 0 false positives and 0 false negatives.
- In reality, no model is 100 % accurate most times.

- The four metrics used to evaluate the performance of a model are:
	- Accuracy: number of correct predictions by the model divided by the total number of predictions => $\frac{TP + TN}{TP + FP + FN + TN}$
	- Precision: equal to true positive (TP) divided by total predicted positive => $\frac{TP}{TP + FP}$ (good metric regarding high costs)
	- Recall or sensitivity: a very useful metric when the cost of a false negative is high => $\frac{TP}{TP + FN}$
	- Specificity: shows how many negative cases were correctly reported as negative => $\frac{TN}{TN + FP}$



### Naive Bayes {#naive-bayes}

- Naive Bayes is a machine learning model that segregates different objects on the basis of certain features of variables (bases on Bayes theorem).
- This classifier predicts the probable occurrence of an event, based on prior knowledge of associated events.
- Example: The probability of the price of a house being high can be assessed with knowledge about the neighbourhood.

- Principle of contingent probability: measures the probability of an event occurring based on something else that has already occurred.
- $P(A \vert B) = \frac{P(B \vert A) \cdot P(A)}{P(B)}$ => Probability of the evidence ($P(B)$) given the hypothesis is true ($P(A)$); $P(B \vert A)$ is the probability of the hypothesis given the evidence is true

- Types of Naive Bayes algorithms:
	1. Gaussian Naive Bayes
	2. Multinomial Naive Bayes
	3. Bernoulli Naive Bayes

- Examples of applications:
	- Identifies the right fit for a job by classifying the skills mentioned in the resume of candidates.
	- Classifies content available in the publishing field, based on information sources such as press, social media, and other sources.
	- Classifies relevant tags in research papers and helps researchers gain access to the right content.
	- Classifies legal papers and identifies illegal documents in any field.
	- Classifies speeches and choice of words of political candidates, and determines the candidates' mentality.
	- Helps determine whether and words or sentences used relates to racial abuse.
	- Classifies degrading sentences or words related to colorism in ad dialogues.
	- Recruitment specialists use Naive Bayes classifiers to check for gender discrimination in hiring practices.



### Applying Naive Bayes classifier {#applying-naive-bayes-classifier}

```python
import numpy as np
import mathplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import make_blobs

# Create X and y variables
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)

# Scatter plot of X and y
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')

# Fit Naive Bayes algorithm to this dataset
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

# Create a few variables to predict
rng = np.random.RandomState(0)
Xnew = [-6, 14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

# Update the 2000 new predictions in the scatter plot
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0], Xnew[:,1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)

# Check for the probability of some of the variables
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
```



### Stochastic Gradient Descent (SGD) {#Stochastic-gradient-descent}

- Stochastic Gradient Descent is a common machine learning algorithm and forms the basis of neural networks.

#### Gradient descent

- Gradient descent is a descending slope that reaches the lowest point on a surface.
- The gradient descent algorithm aims to find the value of 'x' such that 'y' is minimum. => 'y' is the objective function to descend to the lowest point.
- An iterative algorithm that starts from a random point on a function and then travels down its slope in steps until it reaches the lowest point of that function.

**Steps:**
1. Computer the gradient of the function.
2. Pick a random initial value for the parameters (in our case, we would differentiate y with respect to x).
3. Plug in the parameter values and update the function.
4. Calculate the step size for each 'feature': Step size = gradient * learning rate
5. Calculate the new parameters: New params = old params - step size
6. Repeat steps 3 to 5 till the gradient is almost zero.

- The learning rate is a flexibel parameter that influences the algorithm convergence towards the minimum.
- The larger the learning rates, the larger the steps down the slope. So the minimum may get missed. => Use small learning rates.

- Gradient descent algorithm becomes very slow on large datasets. E.g. a set of 10,000 data points 10 features may take as much as 10 million computations to complete the algorithm.


#### Stochastic Gradient Descent (SGD)

- Stochastic means random.
- SGD randomly picks one data point in the whole data set at each iteration. This helps tu hugely reduce the number of computations.
- SGD offers the advantages of speed, efficiency, and ease of implementation.
- Disadvantages of SGD:
	- Due to frequent updates, the steps taken towards the minimum may be very 'noisy'.
	- This may lengthen the time taken to reach the minima.
	- The frequent updates are also computationally expensive.


#### Applications of SGD
- While scaling up, companies find it difficult to keep a check on quality. Machine learning helps to automate the process. SGD helps to tackle low performance issues in the model created.
- SGD also helps to speed up convergence in huge datasets. It is helpful in real-world scenarios that involve large scale data.
- SGD is used in financial modelling to reduce inaccuracies in predictions.




### Applying Stochastic Gradient Descent {#applying-stochastic-gradient-descent}

```python
import numpy as np
import pandas as pd
import mathplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the dataset
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Fit the model in SGD
class CustomPerceptron(object):

	def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):

	self.n_iterations = n_iterations
	self.random_state = random_state
	self.learning_rate = learning_rate

	# Fitting X and y and then adding predicted value
	def fit(self, X, y):
		rgen = np.random.RandomState(self.random_state)
		self.coef_ = rgen.normal(loc=0.0, scale=0.1, size = 1 + X.shape[1])
		for _ in range(self.n_iterations):
			for xi, expected_value in zip(X,y):
				predicted_value = self.predict(xi)
				self.coef_[1:] += self.learning_rate * (expected_value - predicted_value) * xi
				self.coef_[0] += self.learning_rate * (expected_value - predicted_value) * 1

	# Define activation function
	def activation(self, X):
		return np.dot(X, self.coef_[1:]) + self.coef_[0]

	# Define predict function for converting the output
	def predict(self, X):
		output = self.activation(X)
		return np.where(output >= 0.0, 1, 0)

	# Define score function
	def score(self, X, y):
		misclassified_data_count = 0
		for xi, target in zip(X,y):
			output = self.predict(xi)
			if (target != output):
				misclassified_data_count += 1
		total_data_count = len(X)
		self.score_ = (total_data_count - misclassified_data_count) / total_data_count
		return self.score_


# Variable initiation
n_iterations=100
learning_rate=0.01

# Fit the model
prcptrn = CustomPerceptron()
prcptrn.fit(X_train, y_train)

# Print the output
print(prcptrn.score(X_test, y_test))   # 0.906433
prcptrn.score(X_train, y_train)   # 0.929648

# Good result, but it could be optimized
```



### K-nearest neighbour {#k-nearest-neighbour}

- K-nearest neighbour (K-NN) is one of the main machine learning algorithms that supports the supervised learning technique.
- The algorithm assumes a similarity between new and available cases, and puts the new case into the most similar category.
- The K-NN algorithm stores all available data and classifies data points based on similarity measures.
- K-NN algorithms are used for regression and classification problems.
- K-NN is a non-parametric algorithm, and does not make assumptions about the data.
- It stores the dataset at its training phase.
- K-NN classifies new data into a category that is most similar to it.

**Steps:**
1. Load the training with test data.

2. Choose the value of K, i.e. the nearest data points. K can be any integer.

3. For every point within the test data, do the following steps.

	3.1 Calculate space between test data and every row of coaching data using any of the following tactics: Euclidian (most used to calculate the distance), Manhattan, Hamming distance.

	3.2 Support the space value and sort them in ascending order.

	3.3 Choose the highest K rows from the sorted array.

	3.4 Assign a category to the test point

4. End


#### Advantages of K-NN

1. Simple to implement
2. Robust even with noisy training data
3. Simple and effective if the training data is large


#### Disadvantages of K-NN

1. Worth of K must always be determined. => Quite complex.
2. It can be computationally expensive to calculate the space between the info points for all training samples.


#### Application of K-NN:

- It classifies manufactured goods based on: size, grading, priority, location




### Applying K-nearest neighbours {#applying-k-nearest-neighbours}

```python
import pandas as pd
import numpy as np
import mathplotlib.pyplot as plt
import seaborn as sns
%mathplotlib inline

df = pd.read_csv('Social_Network_Ads.csv')

df.head()   # 5 columns: User ID, Gender, Age, EstimatedSalary, Purchased

df.info()   # 400 rows; no null-values

# Check how many people have purchased
df['Purchased'] = value_counts()   # 0: 257; 1: 143

# Create a dummy variable for gender
gender = pd.get_dummies(df['Gender'], drop_first=True)
df = pd.concat([df, gender], axis=1)

# Drop Gender column, since we have converted it into a dummy variable
df.drop(['Gender'], axis=1, inplace=True)

# Create X and y variables
X = df[['Age', 'EstimatedSalary', 'Male']]
y = df['Purchased']

# Standardize the data because its range is different using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

# Transform the features
scaled_features = scaler.transform(X)
scaled_features

# Create a dataframe for transformed data
df_feat = pd.DataFrame(scaled_features, columns=X.columns)
df_feat.head()

# Split the data into training and test datasets with 80:20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.2)

# Import KNeighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)   # Use one neighbour and try to fit it
knn.fit(X_train, y_train)

# Predict and try to evaluate
y_pred = knn.predict(X_test)

# Import classification and confusion metrics
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y_test, y_pred)   # array([49, 4], [2, 25])

print(classification_report(y_test, y_pred))   # accuracy: 0.93; f1-score: 0.89
```




### Decision Tree {#decision-tree}

- Decision Trees (DTs) are a supervised learning non-parametric method used for classification and regression.
- A decision tree is a tree-like structure with nodes, branches, and leaves.
- The nodes represent the place where we want to ask a question or select an attribute (root, decision and leaf nodes).
- The leaves represent the output or class label.
- DTs use such a structure and the associated decision rules.
- The model predicts the value of a target value.

- The root node represents a condition that is split into two results.
- The decision node continues until the tree has no condition left.
- When no further branches arise, it is called leaf node.
- The decision to split into nodes and leaves influences the model's accuracy.
- Decision trees use multiple algorithms to separate a node into two or more sub-nodes.
- The creation of sub-nodes divides the parent node into two parts by increasing the homogeneity of the sub-nodes.
- The purity of the node is directly connected to the target variable.


#### Algorithms that use decision trees

- The type of target variables is an additional input for algorithm selection.
- ID3, C4.5, CART (Cart classification and regression tree), CHAID, MARS


#### Benefits of decision trees

- The tree structure helps to capture the interactions between features within the data.
- Due to their visual representation, it is easier to interpret decision trees than a multi-dimensional hyperplane.   


#### Disadvantages of decision trees

- Decision do not take linear relationships into account.


#### Application of decision trees
- Decision trees are effective in applications with non-linear datasets.
- City planning, Engineering, Law, Business applications



### Applying decision tree {#applying-decision-tree}

```python
import pandas as pd
import numpy as np
import mathplotlib.pyplot as plt
import seaborn as sns
%mathplotlib inline

df = pd.read_csv('balance-scale.data', sep=',')
df.head()   # Five columns: Class Name, Left weight, Left distance, Reight weight, Right distance

df.info()   # 625 rows; no null-values; Class Name is the target variable to be predicted

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X = df.drop('Class Name', axis=1)
y = df['Class Name']

# Split the data into training and test datasets with 70:30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the decision tree model with criterion 'gini'
clf_model = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3, min_samples_leaf=5)
clf_model.fit(X_train, y_train)

# Get the prediction
y_pred = clf_model.predict(X_test)

# Check the metrics
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))   # accuracy: 0.70

# Create a target and plot the decision tree
target = list(df['Class Name'].unique())
feature_names = list(X.columns)

from sklearn.tree import export_tree
r = export_tree(clf_model, feature_names=feature_names)
print(r)   # Plot of decision tree with each child node
```



### Random forest {#random-forest}

- Random forest is a machine learning technique used to solve regression and classification problems.
- It is an ensemble learning technique: combines multiple classifiers to solve a complex problem.
- The random forest algorithm consists of many decision trees.
- The key difference between typical decision trees and the random forest algorithm are:
	- Random forest select a subset of the features.
	- Whereas decision trees consider all the feature splits that are possible.


#### Advantages of random forest

1. Reduced risk of overfitting: In decision trees, data is fixed in leaves and nodes what makes predictions easy. Random forest reduces the risk of overfitting.
2. Provides flexibility: Handles the dataset for classification as well as regression. => most popular method
3. Easy to determine: Determines which variable or feature is more important and has the highest contribution towards the model.


#### Disavantages of random forest

1. Time-consuming process: The algorithm works best with a large amount of data, but it also slows down the processing.
2. Requires more resources: It does not work with a small dataset.
3. More complex: Interpreting the prediction of a forest can get complex.


#### Applications of random forest

- Used across multiple industries.
- In finance it reduces the time to manage a huge dataset.
- It is used to evaluate prices in stock markets or banks to process a loan.
- In large companies, random forest classifiers help to identify the patterns of employee attrition.




### Applying random forest regressor {#applying-random-forest}

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

datasets = pd.read_csv('petrol_consumption.csv')
datasets.head()   # We will predict the petrol consumption.

datasets.info()   # 5 columns, 48 rows. All data types are in numeric. No missing values

# Create X and y variables
X = datasets.iloc[:, 0:4].values
y = datasets.iloc[:, 4].values

# Split the data into training and test datasets with 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Rescale the data using a standard scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Fit the random forest regressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Check the metrics for train and test
print('Train MAE:', mean_absolute_error(y_train, regressor.predict(X_train)))
print('Train RSME:', np.sqrt(mean_squared_error(y_train, regressor.predict(X_train))))
print('R Square:', r2_score(y_train, regressor.predict(X_train)))

print('Test MAE:', mean_absolute_error(y_test, regressor.predict(X_test)))
print('Test RSME:', np.sqrt(mean_squared_error(y_test, regressor.predict(X_test))))
print('R Square:', r2_score(y_test, regressor.predict(X_test)))

# Use cross-validation to minimize the huge differences between train and test metrics.
```



### Boruta {#Boruta}

- Boruta is a feature selection algorithm that works as a wrapper around random forest.
- Predictive modelling is also known as feature selection. It is crucial to get the most important feature.
- Boruta is useful when a dataset contains many features: It eliminates features recursively in each iteration while fitting the random forest model.
- This results in minimum features. This minimizes the error and results in over-pruned input data that throw away some relevant data.
- In contrast, Boruta analyzes all the high and weak features connected to the decision variable. Boruta captures all the features relevant to the dependent or target variable.

#### Applications of Boruta

- Used majorly in the biomedical industry. Example: Specific genes cause various health-related issues => complicated for doctors to find precisely the gene that is causing health-related problems. Boruta helps to determine such genes.



### Automatic feature selection with Boruta {#automatic-feature-selection-with-boruta}

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

URL = "https://raw.githubusercontent.com/Aditya1001001/English-Premier-League/master/pos_modelling_data.csv"
data = pd.read_csv(URL)

data.info()   # 35 columns, 1793 rows; no missing values

# Create X and y variables
X = data.drop('Position', axis=1)
y = data['Position']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit random forest classifiers with all the features
rf_all_features = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
rf_all_features.fit(X_train, y_train)
accuracy_score(y_test, rf_all_features.predict(X_test))   # 0.729805

# Use Boruta for feature selection
rfc = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1)
# Fit the data by passing them in array format
boruta_selector.fit(np.array(X_train), np.array(y_train))   # 31, 1, 2
# Tries to get iterations for everyone of the 100 samples. It shows how many variables are tentative, confirmed, or rejected. Based on that, the ranking will be formed.

# Print the ranking and the number of significant features
print("Ranking:", boruta_selector.ranking_)
print("No. of significant features:", boruta_selector.n_features_)

# See feature names by sorted ranks
selected_rf_features = pd.DataFrame({'Features': list(X_train.columns), 'Ranking': boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking')


# Using important feature selection fit the random forest classifier model
X_imp_train = boruta_selector.transform(np.array(X_train))
X_imp_test = boruta_selector.transform(np.array(X_test))

# Same number of estimators and same depth as above
rf_boruta = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
rf_boruta.fit(X_imp_train, y_train)

accuracy_score(y_test, rf_boruta.predict(X_imp_test))   # 0.732591
```



### Support vector machine {#support-vector-machine}

- Support vector machine (SVM) is a supervised machine learning algorithm used for both classification and regression challenges.
- SVM is mainly used in classification-related problems.
- Plot the data points in an n-dimensional space where n is the number of features in the data. The value of a data point and of a coordinate in the dimension are the same.
- Perform classification using hyperplane.
- The hyperplane helps to differentiate the classes.

- In KNN, a kernel function is used to map lower-dimensional data into higher-dimensional data.


#### Types of support vector machines

- **Linear SVM:** Separates the data in a linear format. If the dataset is separated into two parts using a straight line, **data is linearly separable**.
- **Non-linear SVM:** Used when data is non-linearly separated. If the dataset cannot be separated into two parts using a straight line, **data is non linear**.


#### Applications of SVM

- SVM is used to classify different avatars players used in games, e.g. classify positive and negative avatars in a game.
- SVM algorithm is best used to recommend ads => helps in segregating and recommending ads
- SVM is used to classify cataracts.




### Applying support vector machine {#applying-support-vector-machine}

```python
import pandas as pd
import numpy as np
import seaborn as sns
import mathplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('heart.csv')
df.head()

df.info()   # 14 columns; 303 rows; no missing data

df.describe()

# Plot the connection between 'Age' and 'Cholesterol'
df.plt(kind='scatter', x='Age', y='Chol', alpha=0.5, color='red')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Age-Cholesterol plot')

# Convert the two categorical variables 'chestPain' and 'Thal' to dummy variables
df_new = pd.get_dummies(df, columns = ['chestPain', 'Thal'], drop_first=True)

# Create variables for X and y
X = df_new.drop('AHD', axis=1)
y = df_new.AHD

# Split data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Import GridSearchCV from model selection
from sklearn.model_selection import GridSearchCV

ml = svm.SVC()

para_grid = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

grid = GridSearchCV(ml, para_grid, refit=True, verbose=1, cv=5, n_jobs=-1)

# Fit the training data to the grid search
grid_search = grid.fit(X_train, y_train)   # Fitting 5 folds for each of 16 candidates, totalling 80 fits

print(grid_search.best_params_)   # {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

accuracy = grid_search.best_score_
accuracy   # 0.665561

# Check the confusion matrix for the testing data
y_test_hat = grid.predict(X_test)

confusion_matrix(y_test, y_test_hat)
disp = plot_confusion_matrix(grid, X_test, y_test, cmap=plt.cm.Blues)

# Check the classification report
print(classification_report(y_test, y_test_hat))   # precision: 0.70, recall: 0.59; accuracy: 0.70
```



### Cohen's kappa {#cohen's-kappa}

- Cohen's kappa is a statistical model used to:
	- Measure the reliability of two raters rating the same quantity
	- Identify how frequently the raters are in agreement
- Cohen's kappa is used to compare the predictions done by the machine learning model.
- Software packages and libraries that provide Cohen's kappa: Caret, Weka, Scikit-learn
- Cohen's kappa is calculated based on the confusion matrix. It is complex to interpret as Cohen's kappa takes an imbalance in class distribution to calculate accuracy.
- Formula: $\kappa = \frac{p_0 - p_e}{1 - p_e}$   ($p_0$: accuracy of the model, $p_e$: Measure of the agreement between the model predictions and the actual class values)


#### Key points about Cohen's kappa

- Excellent measure that handles multi-class and imbalanced class problems.
- The value of Cohen's kappe decreases with an increase in the difference between predicted and actual target classes.
- Balanced data gives a higher Cohen's kappa value.
- More informative than overall accuracy when working with unbalanced data.
