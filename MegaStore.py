from sklearn.preprocessing import MinMaxScaler  # scaling each feature to a given range
from sklearn.preprocessing import LabelEncoder  # encode labels
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  # train and test our dataset by spliting
import pandas as pd  # data structures and data analysis tools
import numpy as np  ## linear algebra            #scientific computing in Python(linear algebra...)
import matplotlib.pyplot as plt  # plotting, opens figures on your screen,data processing, CSV...
from warnings import simplefilter
# # ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns  # data visualization library based on matplotlib.
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import math
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
###################################
def displaySpecificRow(boolean):
    if boolean == True:
        inp0 = int(input("enter start index: "))
        inp1 = int(input("enter end index: "))
        return dataset.iloc[inp0:inp1]
    return "\n"


def displaySpecificColumn(boolean):
    if boolean == True:
        inp0 = input("enter column name: ")
        return dataset[inp0]
    return "\n"

###################################
#1.read,visualize:
dataset = pd.read_csv(r"W:\fcis\ml\proj\megastore-regression-dataset.csv")  # to read .csv files

print('Overview of the dataset:\n')
print('Number of rows: ', dataset.shape[0])
print("\nNumber of features:", dataset.shape[1])
print("\nData Features:")
print(dataset.columns.tolist())
print("\nMissing values:", dataset.isnull().sum().values.sum())
print("\nUnique values:")
print(dataset.nunique())  # unique values


inp0 = input("Do yo want Description, (quick describe of numeric data):[y/n] ").lower()
if inp0 == "y":
    #print(dataset.columns, end="\n")
    print("\t \t Description of dataset\n")
    print(dataset.describe())
print(displaySpecificColumn(False))
print(displaySpecificRow(False))

inp1 = input("Do you want reading of top or bottom from dataset before starting:[y/n] ").lower()
if inp1 == "y":
    inp2 = input("top or bottom:[t/b] ").lower()
    if inp2 == "t":
        print(dataset.head())
    else:
        print(dataset.tail())

#2.preprocessing & feature selection ,eng,extraction


print(dataset.isnull().sum())

#if specific column has null cells then we Fill the missing values with  the median value or zeros according to col.
#dataset['col_name'] = dataset['col_name'].fillna(dataset['col_name'].median())
#or
# Remove rows with missing values
dataset.dropna(inplace=True)
#but we dont have


print(dataset.duplicated())
dataset=dataset.drop_duplicates()
# duplicates = dataset[dataset.duplicated()]
# print(duplicates)

# this function split CategoryTree column into 2 cols (main_category, sub_category)
main_cat = []
sub_cat = []


def split_categorytree(df):
    for i in range(len(df)):
        x = eval(df.loc[i, "CategoryTree"])

        main_cat.append(x["MainCategory"])
        sub_cat.append(x["SubCategory"])

    ser1 = pd.Series(main_cat)
    ser2 = pd.Series(sub_cat)
    return ser1, ser2
#
# applying the function on the data
ser1, ser2 = split_categorytree(dataset)
dataset["MainCategory"] = ser1
dataset["SubCategory"] = ser2


# dropping 'CategoryTree' column which is replaced with 2 cols
dataset.drop("CategoryTree", axis=1, inplace=True)

last_col_index = dataset.columns.get_loc("Profit")

# insert "MainCategory" and "SubCategory" before the last column "y"
dataset.insert(loc=last_col_index, column="SubCategory", value=dataset.pop("SubCategory"))
dataset.insert(loc=last_col_index, column="MainCategory", value=dataset.pop("MainCategory"))


#
# import json
# def split_dictionary(dataframe, column_name):
#     # create two new columns by splitting the dictionary values
#     dataframe[column_name] = dataframe[column_name].apply(lambda x: json.loads(x))
#     dataframe['MainCategory'] = dataframe[column_name].apply(lambda x: x['MainCategory'])
#     dataframe['SubCategory'] = dataframe[column_name].apply(lambda x: x['SubCategory'])
#     # drop the original column
#     dataframe.drop(column_name, axis=1, inplace=True)
#     return dataframe
#
# dataset=split_dictionary(dataset,'CategoryTree')
#
#




print(dataset.dtypes)


X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X['Order Date']=pd.to_datetime(X['Order Date'])
X['Order Month']=X['Order Date'].dt.month
X['Order day']=X['Order Date'].dt.day
X['Order year']=X['Order Date'].dt.year
X['Ship Date']=pd.to_datetime(X['Ship Date'])
X['Ship Month']=X['Ship Date'].dt.month
X['Ship day']=X['Ship Date'].dt.day
X['Ship year']=X['Ship Date'].dt.year
X=X.drop(columns=['Ship Date'])
# extract new Feature from current features
X['SalesPerQuantity'] = X['Sales'] / X['Quantity']
X['Order Date'] = (X['Order day'] - X['Ship day'])

xv=X.values


# X = dataset.iloc[:, :18].join(dataset.iloc[:, 19:])
# y=dataset["Profit"]
#X=dataset.drop("Profit",axis=1)



# X=dataset.drop("Profit",axis=1)
# y=dataset["Profit"]
# print(dataset.iloc[:, -1])

serv = ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'CategoryTree', 'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit']
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))  # create subplots with specific rows,cols
for i, item in enumerate(serv):
    if i < 3:
        ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i, 0], rot=0, color='#0009ff')  # calc values
    elif i >= 3 and i < 6:
        ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i - 3, 1], rot=0, color='#9b9c9a')
    elif i < 9:
        ax = dataset[item].value_counts().plot(kind='bar', ax=axes[i - 6, 2], rot=0, color='#ec838a')
    ax.set_title(item)  # title
plt.suptitle('simpleHistogram\n', horizontalalignment="center", fontstyle="normal", fontsize=24, fontfamily="serif")
#plt.show()  # display plots
#no null
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#plt.show()


plt.scatter(xv[:,0], y)
#plt.show()
# X = X.apply(lambda x: x.astype(str))
#X = X.astype(str)
cat=["Order ID","Ship Mode","Customer ID","Customer Name","Segment","Country","City","State","Region","Product ID","Product Name","MainCategory","SubCategory"]
X, xTest, y, yTest = train_test_split(X, y, test_size=0.20)
#num=["Row ID","Postal Code","Sales","Quantity","Discount"]
#cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

#print(cat_cols)
# lb.fit(y)
# X["Order ID"]= lb.transform(X["Order ID"])
# xTest["Order ID"] = lb.transform(xTest["Order ID"])
lb=LabelEncoder()
#lb.fit(X)
for i in cat:
  combined = np.concatenate((X[i], xTest[i]))
  # fit the LabelEncoder on the combined data
  lb.fit(combined)
  X[i]= lb.transform(X[i])
  xTest[i] = lb.transform(xTest[i]) #transform only
scaler = MinMaxScaler()
# for i in num:
#   X[i]= scaler.fit_transform(X[i].values.reshape(-1, 1))


#

num=["Row ID","Postal Code","Sales","Quantity","Discount","Order Date","Order Month","Order day","Order year","Ship Month","Ship day","Ship year","SalesPerQuantity"]
for i in cat:
  combined = np.concatenate((X[i], xTest[i]))
  # fit the LabelEncoder on the combined data
  lb.fit(combined)
  X[i]= lb.transform(X[i])

plt.figure(figsize=(25, 15))
corr = X.corr()
sns.heatmap(corr, annot=True)  # annotation
#plt.show()

X=X.drop("Country",axis=1)
xTest=xTest.drop("Country",axis=1)


def corrl(dataset, threshold):
    cols = []  # set of all names of corelated col
    corl_matrix = dataset.corr()
    for w in range(len(corl_matrix.columns)):
        for ww in range(w):
            if abs(corl_matrix.iloc[w, ww] >= abs(threshold)):  # +,-
                colname = corl_matrix.columns[ww]  # getting name of col
                cols.append(colname)
    return cols

print("high corrlation column: ", corrl(X, -0.85))
ls=corrl(X, -0.85)
for k in ls:
    X = X.drop(f"{k}", axis=1)
    xTest = xTest.drop(f"{k}", axis=1)  # if you drop any columns from the features (X) in your training set (X_train), you should also drop those same columns from the features in your testing set (X_test).

print(X.select_dtypes(include=np.number).var().astype('str'))

#print(X['Sales'].describe())
#X=X.drop("Sales", axis=1)



#print(X.head())

# select the most 7 relavant features
#selector = SelectKBest(f_regression, k=5)


selector = SelectKBest(f_regression, k=17) #drop 3 less than 0.1
X=X.drop(['Order day','Order Date','Ship Month'], axis=1)
xTest=xTest.drop(['Order day','Order Date','Ship Month'], axis=1)

x_new = selector.fit_transform(X,y)

#print(x_new)
# getting the scores

scores = selector.scores_

features_scores= pd.DataFrame({"features":X.columns, "score": scores})

features_scores = features_scores.sort_values(by='score', ascending=False)
print(features_scores)



pca = PCA(n_components=5) # Specify the number of components to keep
X_pca = pca.fit_transform(X)
print(X_pca)

# for i in cat:
#   xTest[i]= lb.transform(xTest[i])
#



xTrain,xVal,yTrain,yVal=train_test_split(X,y,test_size=0.2,random_state=50)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(xTrain, yTrain)#وممكن تشيلو الريدج
test_score = ridge.score(xVal, yVal)
print("R^2 score on test set:", test_score)

model = LinearRegression()

#'cross val' more robust 30k       robust estimate,different splits of the data, dataset is small,model has a large number of hyperparameters to tune,good at unseen
# Split the data into k folds
kfold = KFold(n_splits=5, shuffle=True, random_state=50)
# # Evaluate the model using k-fold cross validation
# model.fit(X, y)
# y_pred = model.predict(X)
# mse = mean_squared_error(y, y_pred)

#'normal val' 30k       simpler ,faster ,single validation set
model.fit(xTrain, yTrain)
y_pred = model.predict(xVal)
mse = mean_squared_error(yVal, y_pred)
print(mse)
results = cross_val_score(model, xVal, yVal, cv=kfold) #X,y are true
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#model.fit(xTest, yTest)
y_pred10 = model.predict(xTest)
mse = mean_squared_error(yTest, y_pred10)

print("loll",mse)


# acc = accuracy_score(yTest, y_pred10)
# print("Validation accuracy:", acc)

#single validation on entire dataset
# Evaluate the model using k-fold cross validation

# acc = accuracy_score(y, y_pred)
# print("Validation accuracy:", acc)
#results = cross_val_score(model, X, y, cv=kfold)
# Print the accuracy score for each fold

# cm = confusion_matrix(yVal, y_pred)
# print(f"Accuracy testing of LogisticRegression algo: {math.sqrt(mse)}")
# sns.heatmap(cm, annot=True, fmt="d")  # decimal
# plt.xlabel("Predicted")
# plt.ylabel("Truth")
# plt.show()


#*******************svm
from sklearn import svm
regressor  = svm.SVR(kernel='rbf',C=0.1)
regressor.fit(xTrain, yTrain)
y_pred1 = regressor.predict(xVal)
mse = mean_squared_error(yVal, y_pred1)
#acc1 = accuracy_score(yVal, y_pred1)
print(mse)

# Specify a range of C values to try
#param_grid = {'C': [0.1, 1, 10, 100]}

# Perform a grid search with 5-fold cross-validation
#grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
#grid_search.fit(X, y)

# Print the best C value and corresponding cross-validation score
# print("Best C value:", grid_search.best_params_['C'])
# print("Cross-validation score:", grid_search.best_score_)




#********************dectree
from sklearn.tree import DecisionTreeRegressor
decTree = DecisionTreeRegressor()
decTree.fit(xTrain, yTrain)
y_pred2 = decTree.predict(xVal)
mse = mean_squared_error(yVal, y_pred2)
#acc1 = accuracy_score(yVal, y_pred1)
print(mse)



###*********************knn
from sklearn.neighbors import KNeighborsRegressor

# Create an instance of the KNeighborsRegressor class
regressor = KNeighborsRegressor(n_neighbors=4)

# Fit the model to your training data
regressor.fit(xTrain, yTrain)

# Make predictions on your test data
y_pred3 = regressor.predict(xVal)
mse = mean_squared_error(yVal, y_pred3)
#acc1 = accuracy_score(yVal, y_pred1)
print(mse)


#*************************gaus
# from sklearn.naive_bayes import GaussianNB
#
# # Create an instance of the GaussianNB class
# regressor = GaussianNB()
#
# # Fit the model to your training data
# regressor.fit(xTrain, yTrain)
#
# # Make predictions on your test data
# y_pred4 = regressor.predict(xVal)
# mse = mean_squared_error(yVal, y_pred4)
# #acc1 = accuracy_score(yVal, y_pred1)
# print(mse)

#*******************************randomforest
from sklearn.ensemble import RandomForestRegressor

# Create an instance of the RandomForestRegressor class
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to your training data
regressor.fit(xTrain, yTrain)

# Make predictions on your test data
y_pred5 = regressor.predict(xVal)
mse = mean_squared_error(yVal, y_pred5)
#acc1 = accuracy_score(yVal, y_pred1)
print(mse)

######################################**poly
from sklearn.preprocessing import PolynomialFeatures
#2==>xx  ,3=>xx, 4=>316877  5=>521374  overfitting
poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(xTrain)

# create a linear regression model
model = LinearRegression()

# fit the model to the data
model.fit(x_poly, yTrain)

# make predictions using the model
x_val_poly = poly.transform(xVal)
y_pred6 = model.predict(x_val_poly)
mse = mean_squared_error(yVal, y_pred6)
#acc1 = accuracy_score(yVal, y_pred1)
print(mse)

# # plot the results
# plt.scatter(x, y)
# plt.plot(x, y_pred, color='red')
# plt.show()



