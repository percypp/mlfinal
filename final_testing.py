import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pydot
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
# import urllib.request
# import re
# import requests
# from bs4 import BeautifulSoup as bs


# For transformations and predictionss
# from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.linear_model import LinearRegression
# from scipy.optimize import curve_fit
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor

# For scoring
# from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse

# For validation
from sklearn.model_selection import train_test_split as split
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')



df.sample(5)
pd.isnull(df).sum() > 0
df.children.fillna(0, inplace=True)
df['total_guests']=df['adults'] + df['children'] + df['babies']
df['total_nights']=df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['arrival_date_month_number'] = df['arrival_date_month'].replace(['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December'], [1,2,3,4,5,6,7,8,9,10,11,12]).astype(str).astype(int)
df['hotel_type'] = df['hotel'].replace(['Resort Hotel', 'City Hotel'], [0,1])
df['country_type'] = df['country']
df.loc[(df['country_type'] != 'PRT'), 'country_type'] = 'International'
df['country_type'] = df['country_type'].replace(['International', 'PRT'], [0,1])
outlier_adr = df.groupby(['adr']).size()
mask= (df['adr']>400) | (df['adr'] <= 0) 
df.loc[mask]
df = df.loc[~mask, :]
mask = (df['total_guests']>=10) | ((df['adults'] == 0) & (df['children'] == 0)) | (df['babies']>=8) 
df = df.loc[~mask, :]



df_full = df[['is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list','required_car_parking_spaces','lead_time','arrival_date_year','hotel_type', 'country_type','arrival_date_month_number', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'total_guests', 'total_nights', 'meal', 'reserved_room_type', 'adr', 'total_of_special_requests']].copy()
df_full.sample(5)
df_full.loc[(df_full['meal'] == 'SC')| (df_full['meal'] == 'Undefined'), 'meal'] = 'SC_Undefined'
meal_order = ['SC_Undefined', 'BB', 'HB', 'FB']
meal_map = dict(zip(meal_order, range(len(meal_order))))
df_full.loc[:, 'meal'] = df_full['meal'].map(meal_map)
reserved_room_type_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']
reserved_room_type_map = dict(zip(reserved_room_type_order, range(len(reserved_room_type_order))))

df_full.loc[:, 'reserved_room_type'] = df_full['reserved_room_type'].map(reserved_room_type_map)
df_full.sample(5)
df_for_lr = df_full[['is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list','required_car_parking_spaces','lead_time','arrival_date_year','hotel_type', 'country_type','arrival_date_month_number', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'total_guests', 'total_nights', 'meal', 'reserved_room_type', 'adr', 'total_of_special_requests']].copy()

def is_family(X):
    if ((X.adults > 0) & (X.children > 0)):
        fam = 1
    elif ((X.adults > 0) & (X.babies > 0)):
        fam = 1
    else:
        fam = 0
    return fam

df_for_lr['is_family'] = df_for_lr.apply(is_family, axis = 1)
def long_stay(X):
    if (X.total_nights > 7):
        stay = 1
    else:
        stay = 0
    return stay

df_for_lr['long_stay'] = df_for_lr.apply(long_stay, axis = 1)
def is_weekend(X):
    if (X.stays_in_weekend_nights != 0):
        we = 1
    else:
        we = 0
    return we

df_for_lr['is_weekend'] = df_for_lr.apply(is_weekend, axis = 1)
df_for_lr = df_for_lr.drop(['adults', 'children', 'babies'], axis = 1)

df_for_lr = df_for_lr.drop(['stays_in_weekend_nights', 'stays_in_week_nights'], axis = 1)

df_for_lr_with_dummies = pd.get_dummies(df_for_lr)
df_for_lr_with_dummies.sample(5)
X = df_for_lr_with_dummies.drop('adr', axis = 1)
y = df_for_lr_with_dummies.adr




X_train, X_test, y_train, y_test = split(X, y, random_state=312150)

# 2. Assign and Fit:

dt_model = DecisionTreeRegressor(max_leaf_nodes=100)

dt_model.fit(X_train, y_train)
dot_data = StringIO()  
export_graphviz(dt_model, out_file=dot_data, feature_names=X.columns, leaves_parallel=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
Image(graph.create_png(), width=750) 
for feature, importance in zip(X.columns, dt_model.feature_importances_):
    print(f'{feature:12}: {importance}')
# 4. Predict:

y_train_pred = dt_model.predict(X_train)

# 5. Visualize:

ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')

# 6. Score:

RMSE_train = np.sqrt(mse(y_train, y_train_pred)).round(3)

# Validate:

y_test_pred = dt_model.predict(X_test)

RMSE_test = np.sqrt(mse(y_test, y_test_pred)).round(3)

print('Decision Tree train RMSE is ', RMSE_train)
print('Decision Tree test RMSE is ', RMSE_test)


pd.isnull(df_test).sum() > 0
df_test.children.fillna(0, inplace=True)
df_test['total_guests']=df_test['adults'] + df_test['children'] + df_test['babies']
df_test['total_nights']=df_test['stays_in_weekend_nights'] + df_test['stays_in_week_nights']
df_test['arrival_date_month_number'] = df_test['arrival_date_month'].replace(['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December'], [1,2,3,4,5,6,7,8,9,10,11,12]).astype(str).astype(int)
df_test['hotel_type'] = df_test['hotel'].replace(['Resort Hotel', 'City Hotel'], [0,1])
df_test['country_type'] = df_test['country']
df_test.loc[(df_test['country_type'] != 'PRT'), 'country_type'] = 'International'
df_test['country_type'] = df_test['country_type'].replace(['International', 'PRT'], [0,1])

mask1 = (df_test['total_guests']>=10) | ((df_test['adults'] == 0) & (df_test['children'] == 0)) | (df_test['babies']>=8) 
df_test = df_test.loc[~mask1, :]



df_test_full = df_test[['is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list','required_car_parking_spaces','lead_time','arrival_date_year','hotel_type', 'country_type','arrival_date_month_number', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'total_guests', 'total_nights', 'meal', 'reserved_room_type', 'total_of_special_requests']].copy()
df_test_full.sample(5)
df_test_full.loc[(df_test_full['meal'] == 'SC')| (df_test_full['meal'] == 'Undefined'), 'meal'] = 'SC_Undefined'
meal_order = ['SC_Undefined', 'BB', 'HB', 'FB']
meal_map1 = dict(zip(meal_order, range(len(meal_order))))
df_test_full.loc[:, 'meal'] = df_test_full['meal'].map(meal_map1)
reserved_room_type_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L']
reserved_room_type_map = dict(zip(reserved_room_type_order, range(len(reserved_room_type_order))))

df_test_full.loc[:, 'reserved_room_type'] = df_test_full['reserved_room_type'].map(reserved_room_type_map)
df_test_full.sample(5)
df_test_for_lr = df_test_full[['is_repeated_guest','previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list','required_car_parking_spaces','lead_time','arrival_date_year','hotel_type', 'country_type','arrival_date_month_number', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'total_guests', 'total_nights', 'meal', 'reserved_room_type', 'total_of_special_requests']].copy()

def is_family(X):
    if ((X.adults > 0) & (X.children > 0)):
        fam = 1
    elif ((X.adults > 0) & (X.babies > 0)):
        fam = 1
    else:
        fam = 0
    return fam

df_test_for_lr['is_family'] = df_test_for_lr.apply(is_family, axis = 1)
def long_stay(X):
    if (X.total_nights > 7):
        stay = 1
    else:
        stay = 0
    return stay

df_test_for_lr['long_stay'] = df_test_for_lr.apply(long_stay, axis = 1)
def is_weekend(X):
    if (X.stays_in_weekend_nights != 0):
        we = 1
    else:
        we = 0
    return we

df_test_for_lr['is_weekend'] = df_test_for_lr.apply(is_weekend, axis = 1)
df_test_for_lr = df_test_for_lr.drop(['adults', 'children', 'babies'], axis = 1)

df_test_for_lr = df_test_for_lr.drop(['stays_in_weekend_nights', 'stays_in_week_nights'], axis = 1)

df_test_for_lr_with_dummies = pd.get_dummies(df_test_for_lr)

df_test = df_test_for_lr_with_dummies
result = dt_model.predict(df_test)
result1 = pd.DataFrame(result)
df_test1 = pd.read_csv('test.csv')
result2 = pd.concat([df_test1,result1],axis=1)
result2.to_csv('test_adr.csv', mode='a', index=False)


