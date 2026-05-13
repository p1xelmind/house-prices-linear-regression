import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

train = pd.read_csv('data/train.csv')
train['SalePrice_log'] = np.log1p(train['SalePrice'])

ordinal_mapping = {
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['NoBasement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['NoBasement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtExposure': ['NoBasement', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['NoBasement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['NoBasement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'FireplaceQu': ['NoFireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageFinish': ['NoGarage', 'Unf', 'RFn', 'Fin'],
    'GarageQual': ['NoGarage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['NoGarage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PoolQC': ['NoPool', 'Fa', 'TA', 'Gd', 'Ex'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'LandSlope': ['Sev', 'Mod', 'Gtl'],
    'PavedDrive': ['N', 'P', 'Y'],
    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'Fence': ['NoFence', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
}

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.to_list()
numeric_features = [col for col in numeric_features if col not in ['Id', 'SalePrice', 'SalePrice_log',
                                                                   'GarageYrBlt', 'MasVnrArea']]

numeric_features_zero = ['GarageYrBlt', 'MasVnrArea']

ordinal_features = list(ordinal_mapping.keys())

categorical_features = train.select_dtypes(include=['string', 'object']).columns.to_list()
categorical_features = [col for col in categorical_features if col not in ordinal_features]
categorical_features_exceptions = ['MiscFeature', 'Alley', 'MasVnrType', 'GarageType']
categorical_features = [col for col in categorical_features if col not in categorical_features_exceptions]

categories_list = [ordinal_mapping[col] for col in ordinal_features]

ordinal_features_exceptions = ['PoolQC', 'Fence', 'FireplaceQu', 'GarageType', 'BsmtExposure', 
                               'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']

ordinal_features_new = [col for col in ordinal_features if col not in ordinal_features_exceptions]
categories_list_new = [ordinal_mapping[col] for col in ordinal_features_new]

num_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

numeric_zero_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

pool_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoPool')),
    ('ord', OrdinalEncoder(categories=[ordinal_mapping['PoolQC']],
                           handle_unknown='use_encoded_value',
                           unknown_value=-1))
])

misc_feature_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoMiscFeature')),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

alley_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoAlley')),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

fence_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoFence')),
    ('ord', OrdinalEncoder(categories=[ordinal_mapping['Fence']],
                           handle_unknown='use_encoded_value',
                           unknown_value=-1))
])

mas_vnr_type_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoMasVnrType')),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

fireplace_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoFireplaceQu')),
    ('ord', OrdinalEncoder(categories=[ordinal_mapping['FireplaceQu']],
                           handle_unknown='use_encoded_value',
                           unknown_value=-1))
])

garage_str_cols = ['GarageFinish', 'GarageQual', 'GarageCond']
garage_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoGarage')),
    ('ord', OrdinalEncoder(categories=[ordinal_mapping[col] for col in garage_str_cols],
                           handle_unknown='use_encoded_value',
                           unknown_value=-1))
])

garage_type_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoGarage')),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

bsmt_str_cols = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
bsmt_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='NoBasement')),
    ('ord', OrdinalEncoder(categories=[ordinal_mapping[col] for col in bsmt_str_cols],
                           handle_unknown='use_encoded_value',
                           unknown_value=-1))
])

categorical_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

ordinal_pipe = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(categories=categories_list_new,
                           handle_unknown='use_encoded_value',
                           unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_pipe, numeric_features),
        ('num_zero', numeric_zero_pipe, numeric_features_zero),
        ('pool', pool_pipe, ['PoolQC']),
        ('misc_feat', misc_feature_pipe, ['MiscFeature']),
        ('alley', alley_pipe, ['Alley']),
        ('fence', fence_pipe, ['Fence']),
        ('mas_vnr_type', mas_vnr_type_pipe, ['MasVnrType']),
        ('fireplace', fireplace_pipe, ['FireplaceQu']),
        ('garage', garage_pipe, garage_str_cols),
        ('garage_type', garage_type_pipe, ['GarageType']),
        ('bsmt', bsmt_pipe, bsmt_str_cols),
        ('cat', categorical_pipe, categorical_features),
        ('ord_simple', ordinal_pipe, ordinal_features_new)
    ]
)

full_model_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.000543))
])

X = train.drop(columns=['Id', 'SalePrice', 'SalePrice_log'])
y = train['SalePrice_log']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

full_model_pipeline.fit(X_train, y_train)

y_pred = full_model_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'RMSE on test dataset: {rmse:.4f}')

full_model_pipeline.fit(X, y)
joblib.dump(full_model_pipeline, 'model.joblib')