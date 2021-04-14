## Introduction

Regression and classification algorithms are two types divided into supervised machine learning algorithms. For supervised learning and unsupervised learning, the difference between them is that whether or not we label the predictors. Specifically regression is to predict the label which is a numeric value but the classification tries to predict the label into probability values, say, true or false(0,1), simply put, when predicting one class of a feature as true, the other classes are all the false, this is a general scenario. In this article, the fundementals of regression will be introduced. Let's start!

## 1. Explore the Data

In this part we will introduce regression on a study for a bicycle sharing scheme was collected and used to predict the number of rentals based on seasonality and weather conditions. The data used is derived from Capital Bikeshare.

Let's load the bicycle sharing data:
```python
import pandas as pd

# load the training dataset
bike_data = pd.read_csv('data/daily-bike-share.csv')
bike_data.head()
```

There are 14 features, in this dataset, rentals is the number of bicyle rentals recordered, as the label that we need to predict.

![image](https://user-images.githubusercontent.com/71245576/114741208-88831500-9d18-11eb-9003-d2cddde432b0.png)

After loading, perform some engineering for preprocessing. Honestly, data preprocessing, including scale type conversion, data scaling, normalization, feature selection and filtering(i.e.,embedding and wrappers) are really desvered to hardly learn for data scientists, here we just perform some assignments for consistency and data quality.

Indexing the "dteday" to the day of the month from 1 to 31:
```python
bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
bike_data.head(32)
```

Here we are:

![image](https://user-images.githubusercontent.com/71245576/114745844-16f99580-9d1d-11eb-90fd-a68d19376b59.png)

Let's grab some key descriptive statistics:

```python
numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
bike_data[numeric_features + ['rentals']].describe()
```

There are 731 rows and no missing values for temp, atemp, hum, windspeed and rentals. To look at our label, the mean of rentals is 488, and it has a comparativly huge standard deviation, it indicates the number of rentals may be various per day.

![image](https://user-images.githubusercontent.com/71245576/114743170-69858280-9d1a-11eb-821e-9d136456fb6b.png)

To get the label column:
```python
import pandas as pd
import matplotlib.pyplot as plt

# This ensures plots are displayed inline in the Jupyter notebook
%matplotlib inline

# Get the label column
label = bike_data['rentals']
```
Plot the histogram and boxplot for the label

```python
# Create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

# Plot the histogram   
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Add lines for the mean, median, and mode
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Plot the boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')

# Add a title to the Figure
fig.suptitle('Rental Distribution')

# Show the figure
fig.show()
```

See the result:

![image](https://user-images.githubusercontent.com/71245576/114745960-34c6fa80-9d1d-11eb-9816-59393790dca5.png)

It shows that the number of rentals daily ranging from 0 to 3,400. The mean of daily rentals is closer to the lower end of the range, most of the data between 0 and around 2,200. Not hard to know that some outliers are really big so that compensate for the mean of the number of daily rentals.

Before the rough analysis of the label, let's take a look at each numeric feature.

```python
# Plot a histogram for each numeric feature
for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    feature.hist(bins=100, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)
plt.show()
```
![image](https://user-images.githubusercontent.com/71245576/114744794-0f85bc80-9d1c-11eb-9372-8c71944997f4.png)

![image](https://user-images.githubusercontent.com/71245576/114744818-17ddf780-9d1c-11eb-98f8-84ea7c9c1ab6.png)

![image](https://user-images.githubusercontent.com/71245576/114744855-1f9d9c00-9d1c-11eb-9ff6-099912a6f706.png)

![image](https://user-images.githubusercontent.com/71245576/114744889-26c4aa00-9d1c-11eb-91d3-d57eb24c6d48.png)

The tutorial tells me that it thinks the numeric features seem to be more normaly distributed, but I do not think so. You can see the figures showed above, it makes some sense to say the hum and windspeed are normally distributed even though there are a little bit skewness, which does not matter for estimation. However, the features, atemp and temp are more like "M", we can discuss how to fit or convert them to a well-fitted distribution, but now I consider it is less like a Gaussian distribution.

So, what about the categorical features? Let's plot bar charts to grab some intuition.

```python
import numpy as np

# plot a bar plot for each categorical feature count
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

for col in categorical_features:
    counts = bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
plt.show()
```
See the result:

![image](https://user-images.githubusercontent.com/71245576/114746311-97b89180-9d1d-11eb-837f-a6b8a7071569.png)

![image](https://user-images.githubusercontent.com/71245576/114746366-a8690780-9d1d-11eb-915b-d3849ad03d1f.png)

![image](https://user-images.githubusercontent.com/71245576/114746387-b159d900-9d1d-11eb-954e-c11f221f1776.png)

![image](https://user-images.githubusercontent.com/71245576/114746431-bdde3180-9d1d-11eb-9e07-e6604cd50e3d.png)

![image](https://user-images.githubusercontent.com/71245576/114746460-ca628a00-9d1d-11eb-93d3-d86d197e4351.png)

![image](https://user-images.githubusercontent.com/71245576/114746493-d4848880-9d1d-11eb-9d80-c227a3f66d55.png)

![image](https://user-images.githubusercontent.com/71245576/114746517-dcdcc380-9d1d-11eb-806b-71c64d88034e.png)

except holiday, workingday, weathersit, others are uniformed(in uniform distribution).

We can also calculate the correlation to find the relationship:
```python
for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    label = bike_data['rentals']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Bike Rentals')
    ax.set_title('rentals vs ' + col + '- correlation: ' + str(correlation))
plt.show()
```

The rentals vs temp and atemp are positively correlated, but the rentals vs hum and windspeed are negatively correlated, the relationship between rentals and humidity is not significant.

![image](https://user-images.githubusercontent.com/71245576/114753286-490ef580-9d25-11eb-8014-7abffb6f1a96.png)

![image](https://user-images.githubusercontent.com/71245576/114753308-51ffc700-9d25-11eb-8d4c-2d707b74f3a9.png)

![image](https://user-images.githubusercontent.com/71245576/114753340-5c21c580-9d25-11eb-9802-f33a548cbdc1.png)

![image](https://user-images.githubusercontent.com/71245576/114753368-63e16a00-9d25-11eb-84bd-cc33a3ebb58c.png)

Now let's compare the categorical features to the label, we do it by performing for each category.

```python
# plot a boxplot for the label by each categorical feature
for col in categorical_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    bike_data.boxplot(column = 'rentals', by = col, ax = ax)
    ax.set_title('Label by ' + col)
    ax.set_ylabel("Bike Rentals")
plt.show()
```

The firgures are very big so that I would not show it here, feel free try execute the statements. Personally the point that I would like to say is that there is a difference in the distribution of rentals on weekday 0 or 6 and those during thw weekday 1 to 5. Meanwhile, there is big difference between different seasons like comparing summer and fall to spring and winter months.


## 2. Regression modelling

Hitherto we understand a sort of about our data, it is time to train a regression model which matches our topic.

### 2.1 Feature separation 

The first thing we need to do is that separate the data from original data, the data set would contain the features that we selected, including season, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, and windspeed.

```python
X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')
```
### 2.2 Data splitting

Now split the data into training dataset and validation data set. What should be noticed is that sometimes we split the data into three parts: training data set, validation data set and testing data set. The general proportions of them is 6:2:2 or you can set any proportions for them, but should take care of the volume of the data set and the methods that you used to split, there are several randomization methods to help you.

```python
from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))
```
In the tutorial, the data splitted to 70% of training data set and 30% of test data set. Training set accordingly has 511 rows and the test set has 220 rows.

### 2.3 Modeling and evaluation

To go ahead, train the linear regression model.
```python
# Train the model
from sklearn.linear_model import LinearRegression

# Fit a linear regression model on the training set
model = LinearRegression().fit(X_train, y_train)
print (model)
```
Compare the predicted values and actual values of the label
```python
import numpy as np

predictions = model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels   : ' ,y_test[:10])
```

There are the indications that show the specific values of the predicted and actual labels, but how good the model represented? obviously we cannot directly get the performance on these values.

![image](https://user-images.githubusercontent.com/71245576/114773149-abbfbb80-9d3c-11eb-8269-d2648458a82e.png)

So now we try to plot the regression on a scatter plot:

```python
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```
There's a definite diagonal trend, and the intersections of the predicted and actual values are generally following the path of the trend line; but there's a fair amount of difference between the ideal function represented by the line and the results. This variance represents the residuals of the model - in other words, the difference between the label predicted when the model applies the coefficients it learned during training to the validation data, and the actual value of the validation label. These residuals when evaluated from the validation data indicate the expected level of error when the model is used with new data for which the label is unknown.

Let's use some metrics to evaluate the model:

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)
print("R2:", r2)
```

MSE, RMSE, R-squared are three metrics widely used on least square regression. Mean Square Error, MSE is the mean of the squared differences between predicted and actual values. Root Mean Square Error, RMSE, is the square root of the MSE. Coefficient of Determination, also called R-squared, shows that the higher this value, the better the fit of the model. Please see the metrics of this model:

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)
print("R2:", r2)
```
The MSE, RMSE, R2 shows below:

![image](https://user-images.githubusercontent.com/71245576/114773789-7b2c5180-9d3d-11eb-9b47-78ecda111a4d.png)

## 3. Experiment with other algorithms

In regression field, there are not only linear regressions, but like the tree based algorithms and ensemble algorithms. For linear algorithm, there are also many different types of algorithms. For being a data scientist, this is a very simple course to learn and reason the models and make deliverables to business objects. Now, let's try to look at some other algorithms, the first is Lasso algorithm. 

Lasso algorithm is a type of linear algorithms. We will discuss it in further article very specifically.

## 3.1 Lasso regression

Let's train the lasso regression in this bicycle sharing project:

```python
from sklearn.linear_model import Lasso

# Fit a lasso model on the training set
model = Lasso().fit(X_train, y_train)
print (model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```
The result shows that the MSE of the lasso is 201155.7, the RMSE is 448 and the R2 is 0.605. See the regression in the scatter plot:

![image](https://user-images.githubusercontent.com/71245576/114774538-4a005100-9d3e-11eb-8435-8ccb3c083f86.png)

## 3.2 Decision tree

Decision tree is a typical model of tree-based models, sometimes it is widely used in the overfitting problem-solving process as well as feature selection in which the decision trees are as the embedding to help select features. 

Let's train a decision tree regression model in the bike rental data and evaluate it now:

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

# Train the model
model = DecisionTreeRegressor().fit(X_train, y_train)
print (model, "\n")

# Visualize the model tree
tree = export_text(model)
print(tree)

# Evaluate the model using the test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```

See the resulting outcome:

![image](https://user-images.githubusercontent.com/71245576/114775016-ceeb6a80-9d3e-11eb-96cb-fc073602b0dc.png)

It seems that the decision tree has not improved over the linear model.

### 3.3 Ensemble random forest

Emsemble learning is widely used in many fields of scenarios in machine learning. From my experience, I have ever performed ensemble decision forests, emsemble random forests and so on in ensemble learning background. The pivotal point of ensemble learning is that it would combine multiple estimators(models) and weight them and finally produce an optimal model, the methods to produce the optimum are bootsting and bagging.

Now we are going to train a random forest in this data set, random forest applies an averaging function to multiple decision tree models:

```python
from sklearn.ensemble import RandomForestRegressor

# Train the model
model = RandomForestRegressor().fit(X_train, y_train)
print (model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```

Let's see the result:

![image](https://user-images.githubusercontent.com/71245576/114775553-6ea8f880-9d3f-11eb-87d2-95715f51a9d4.png)

You want better performance? try boosting ensemble learning now, we could use a gradient boosting estimator, which tries to incrementally reduce the losee in the model.

```python
# Train the model
from sklearn.ensemble import GradientBoostingRegressor

# Fit a lasso model on the training set
model = GradientBoostingRegressor().fit(X_train, y_train)
print (model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```
The model performed better than ever.

![image](https://user-images.githubusercontent.com/71245576/114775893-d5c6ad00-9d3f-11eb-9071-b54e096321ee.png)

### 3.4 Hyperparameter optimization

We know that parameters are values that can be determined from data, the hyperparameters are that values that we can specify to affect the behavior of a training algorithm, whose value cannot be estimated from data and seemly like a configuration that is external to the model.

Let's try to use a grid search approach to try combinations from a graid of possible values for the learning_rate and n_estimators hyperparameters of the gradient boosting regressor estimator.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score

# Use a Gradient Boosting algorithm
alg = GradientBoostingRegressor()

# Try these hyperparameter values
params = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150]
 }

# Find the best hyperparameter combination to optimize the R2 metric
score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

# Get the best model
model=gridsearch.best_estimator_
print(model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()
```

The result:

![image](https://user-images.githubusercontent.com/71245576/114794079-b9d00500-9d59-11eb-9a0f-d088c2236b65.png)

## 4. Reference

Train and evaluate regression models, retrieved from https://docs.microsoft.com/en-us/learn/modules/train-evaluate-regression-models/


