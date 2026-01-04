import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

 

housing = pd.read_csv("housing.csv")
 

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)
 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

housing = strat_train_set.copy()
 
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
print(housing.head())

num_attrb=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attrb=["ocean_proximity"]

my_num=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

my_cat=Pipeline([
    ("encoder",OneHotEncoder())
])

full_pipeline=ColumnTransformer([
    ("num",my_num,num_attrb),
    ("cat",my_cat,cat_attrb)
])

housing_pre=full_pipeline.fit_transform(housing)
print(housing_pre.shape)
print(housing_pre)
model=RandomForestRegressor()
model.fit(housing_pre,housing_labels)
prediction=model.predict(housing_pre)
print(prediction)

