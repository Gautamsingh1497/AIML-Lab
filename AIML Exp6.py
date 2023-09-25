#Import all the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#read csv file
health_data=pd.read_csv('brain_stroke.csv',index_col=0,na_values="Unknown")
#print csv file
health_data
#prints information about the DataFrame
print(health_data.info)
# Create a copy of this dataframe and remove duplicate rows while keeping the first occurrence and modification to be made in copied data itself.
stroke=health_data.copy()
stroke.drop_duplicates(keep='first',inplace=True)
#Check for null values
stroke.isnull().sum()
#Drop null values
stroke_omit=stroke.dropna(axis=0)
stroke_omit
#Convert categorical values into continuous variables
stroke_omit=pd.get_dummies(stroke_omit,drop_first=True)
stroke_omit
#Import following functions from sci-kit learn module
from sklearn.model_selection import train_test_split #split data into train and test dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #metric to evaluate regression model
#Create x1 as independent variable and y1 as dependent variable
x1= stroke_omit.drop(['age'], axis='columns', inplace=False)
y1= stroke_omit['age']
#split the data into training and testing sets and print their shape
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Creating an instance of a Linear Regression model using scikit-learn as model will estimate an intercept term
lgr=LinearRegression(fit_intercept=True)
#Trains the model by adjusting its parameters to minimize the difference
model_lin1=lgr.fit(X_train, y_train)
#array containing the model's predictions for the target variable based on the features in
X_test
stroke_pred = lgr.predict(X_test)
stroke_pred
#Evaluate root mean squared error
lin_mse1 = mean_squared_error(y_test, stroke_pred)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)
#generates a scatter plot where the x-axis represents the predicted values and the y-axis represents the residuals.
residuals=y_test-stroke_pred
sns.regplot(x=stroke_pred, y=residuals, scatter=True, fit_reg=False)
#give information on residual
residuals.describe()


#read heart dataset csv file
heart_data=pd.read_csv('heart.csv')

#print information on heart dataset csv file
print(heart_data.info)
#check for null values
heart_data.isnull().sum()
#Extracting features and target values from dataframe
x2 = heart_data.iloc[:,0:-1].values
y2 = heart_data.iloc[:, -1].values

#Print shape of training and testing datasets
x_train, x_test, Y_train, Y_test = train_test_split(x2, y2, test_size = 0.30, random_state=4)
print(x_train.shape, x_test.shape, Y_train.shape, Y_test.shape)
#perform feature scaling operation to standardize features by removing the mean and scaling to unit variance.
from sklearn.preprocessing import StandardScaler
s1 = StandardScaler()
x_train = s1.fit_transform(x_train)
x_test= s1.transform (x_test)
#Import following function from sci-kit learn module and adjust its parameter to perform prediction
from sklearn.linear_model import LogisticRegression
model_logistic = LogisticRegression()
model_logistic.fit(x_train, Y_train)
Y_pred=model_logistic.predict(x_test)

#get the decision function scores for a given set of samples
y_pred_logistic = model_logistic.decision_function (x_test)
Y_pred
#generates an ROC curve plot for a logistic regression model, including the AUC value in the legend
from sklearn.metrics import roc_curve, auc
logistic_fpr, logistic_tpr, threshold = roc_curve (Y_test, y_pred_logistic)
auc_logistic = auc (logistic_fpr, logistic_tpr)
plt.figure(figsize=(5, 5), dpi=100)
plt.plot (logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()