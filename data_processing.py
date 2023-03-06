import re
import numpy as np
import pandas as pd

SEED = 0
PATTERN = "\t| "
INCORRECT_NAMES = {
    "Bedroom": "BedroomAbvGr",
    "Kitchen": "KitchenAbvGr",
}
    
##############################################
##############################################
##############################################  

def extract_data_descr(filepath):
    data_descr = {}
    with open(filepath) as f:
        key = None
        val = []
        for line in f.readlines():
            line = line.strip()
            if line:
                line = re.split(PATTERN, line)[0]
                if key and not line.endswith(":"):
                    val.append(line)
                elif key:
                    data_descr[key] = val
                    val = []
                    key = line[:-1]
                else:
                    # for first iteration
                    key = line[:-1]

        data_descr[key] = val
      
    for old, new in INCORRECT_NAMES.items():
        data_descr[new] = data_descr.pop(old)
        
    return data_descr

##############################################
##############################################
##############################################

def check_missing_values(train_df, test_df):
    print("="*40)
    print(f"(TRAIN) Number of missing values:")
    print("="*40)
    print(train_df.isna().sum()[train_df.isna().sum()>0])
    print("="*40)
    print(f"(TEST) Number of missing values:")
    print("="*40)
    print(test_df.isna().sum()[test_df.isna().sum()>0])

##############################################
##############################################
##############################################

def fill_with_constant(train_df, test_df, na_col):    
    train_data = train_df.copy()
    test_data = test_df.copy()
    
    # fill all specified columns except dates
    fill_col = [
        col for col in na_col
        if all(not date in col for date in ["Yr", "Year", "Mo"])
    ]
    fill_values = train_data[fill_col].dtypes.apply(
        lambda x: 0 if np.issubdtype(x, np.number) else "NA"
    ).to_dict()
    
    train_data[~train_data[na_col].any(axis=1)] = \
    train_data[~train_data[na_col].any(axis=1)].fillna(fill_values)
    test_data[~test_data[na_col].any(axis=1)] = \
    test_data[~test_data[na_col].any(axis=1)].fillna(fill_values)
    
    return train_data, test_data

##############################################
##############################################
##############################################

def fill_with_estimate(train_df, test_df, related_col):
    train_data = train_df.copy()
    test_data = test_df.copy()
    
    fill_fn = lambda x: x.median() if np.issubdtype(x, np.number) else x.mode()[0]
   
    # remove houses without the corresponding feature before estimating the fill values
    fill_exclude_na = {}
    for feat, cols in related_col.items():
        hse_no_feat_idx = (train_data[cols]=="NA").sum(axis=1).astype(bool)
        fill_exclude_na.update(train_data[~hse_no_feat_idx][cols].apply(fill_fn).to_dict())
     
    # other columns
    other_col = [
        col for col in train_data.columns
        if not any(feat in col for feat in related_col.keys())
    ]
    fill_other = train_data[other_col].apply(fill_fn).to_dict()
    
    fill_values = {**fill_exclude_na, **fill_other}
    train_data = train_data.fillna(fill_values).astype(train_data.dtypes.to_dict())
    test_data = test_data.fillna(fill_values).astype(train_data.dtypes.to_dict())
    
    return train_data, test_data

##############################################
##############################################
##############################################

class RegressionImputer:
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        X = X.copy()
        y = y.copy()
        
        X, y = X[~( X.isna()|y.isna() )], y[~( X.isna()|y.isna() )]
        X = X.loc[y.index]
        y = np.log1p(y.values)
        X = np.log1p(X.values.reshape(-1, 1))
        self.model.fit(X, y)
        
        return self
    
    def transform(self, X, y):
        if y.isna().sum() == 0:
            return y
        
        X = X.copy()
        y = y.copy()
        
        idx = y[y.isna()].index
        X = X.loc[idx]
        X = np.log1p(X.values.reshape(-1, 1))
        y_pred = self.model.predict(X)
        y[y.isna()] = np.expm1(y_pred)
        
        return y

##############################################
##############################################
##############################################

def dummy_encoder(train_df, test_df):    
    train_data = train_df.copy() 
    test_data = test_df.copy()
    
    # combine training and testing data to ensure all possible categories are present
    train_data["Train"] = 1
    test_data["Train"] = 0
    data = pd.concat([train_data, test_data], ignore_index=True)
    cat_data = data.select_dtypes(include=object).astype(str)
    if len(cat_data.columns) == 0:
        raise Exception("Missing categorical columns in training/testing data")
        
    # convert categorical columns to dummy columns
    dummies_df = pd.get_dummies(cat_data, drop_first=True)
    data = pd.concat([dummies_df, data.select_dtypes(exclude=object)], axis=1)
    train_data = data[data.Train==1].drop(columns=["Train"])
    test_data = data[data.Train==0].drop(columns=["Train"])
    
    # remove constant columns in training data
    const_col = train_data.nunique()[train_data.nunique()==1].index.tolist()
    train_data = train_data.drop(columns=const_col)
    test_data = test_data.drop(columns=const_col)
    
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)
    
    print("Dimensions of training data from", train_df.shape, "to", train_data.shape)
    print("Dimensions of testing data from", test_df.shape, "to", test_data.shape)
    
    return train_data, test_data