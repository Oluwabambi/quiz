import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from sklearn.linear_model import Lasso, Ridge


df = pd.read_csv('energydata_complete.csv')
new_df = df.drop(columns = ['date','lights'])

one_df = new_df[['T2','T6','Appliances']]

scaler = MinMaxScaler()
norm_df = pd.DataFrame(scaler.fit_transform(new_df), columns=new_df.columns)
features = norm_df.drop(columns = 'Appliances')
target = norm_df['Appliances']

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
predict = linear_model.predict(x_test)

r2_score = r2_score(y_test, predict)
round(r2_score, 2)

mae = mean_absolute_error(y_test, predict)
round(mae, 2)

rss = np.sum(np.square(y_test - predict))
round(rss, 2)

rmse = np.sqrt(mean_squared_error(y_test, predict))
round(rmse, 3)

def get_weights_df(model, feat, col_name):
  #this function returns the weight of every feature
  weights = pd.Series(model.coef_, feat.columns).sort_values()
  weights_df = pd.DataFrame(weights).reset_index()
  weights_df.columns = ['Features', col_name]
  weights_df[col_name].round(3)
  return weights_df

linear_model_weights = get_weights_df(linear_model, x_train, 'Linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_df = get_weights_df(lasso_reg, x_train, 'Lasso_weight')

final_weights = pd.merge(linear_model_weights, ridge_weights_df, on='Features')
final_weights = pd.merge(final_weights, lasso_weights_df, on='Features')
final_weights
