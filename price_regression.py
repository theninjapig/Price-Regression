import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import  mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor



def select_features(dfp):
    """
    Selects categorical features and hashes them from strings to ints
    """
    df =  dfp[['5', '7', '12']].copy()  
    df.columns=['type', 'duration','location']
    for col in df.columns:
        strings = df[col].unique()
        if col != "location":
            mapper = dict( zip(strings, range(len(strings))) )
            df[col].replace(mapper, inplace=True)
        else:
            df[col] = df[col].str.contains("LONDON").astype(int)
    return df

  
def run_model(Xp_train=None,y_train=None,Xp_test=None,y_test=None,mt='bl', params={}):
    """
    Fits and evaluates the model specfied by string paramater mt
    """
    models = {  
                "bl": lambda : None,
                "lr": LinearRegression,
                "dtr": DecisionTreeRegressor,
                "gbr": GradientBoostingRegressor
    }

    # Select Model, returns None for baseline
    model = models[mt](**params)

    # Fit Model and make predictions
    if mt != 'bl': # If model not baseline
        # Dummy encoding , (each category of n levels or attributes is converted into n-1 dichotomous variables)
        X_train = pd.get_dummies(Xp_train, columns=['type', 'duration','location'],drop_first=True)
        X_test = pd.get_dummies(Xp_test, columns=['type', 'duration','location'],drop_first=True)
        model.fit(X_train,y_train)
        # Make Predictions
        y_pred = model.predict(X_test)
        #print X_test.shape
        y_predtr = model.predict(X_train)       
    else: # Compute baseline
        y_pred = np.median(y_train).repeat(len(y_test))
        y_predtr = np.median(y_train).repeat(len(y_train))
        
      

    # Report metrics
    print("Model name: %s" % (model.__class__.__name__ if model else "Price Median Baseline"))
    if mt != 'bl':
        print("hyper-parameters: " + ", ".join( "{0}: {1}".format(k,v)  for (k,v) in params.items() ) )
    print("Mean absolute error training set: %.2f" % mean_absolute_error(y_train, y_predtr))    
    print("Mean absolute error testing set: %.2f \n" % mean_absolute_error(y_test, y_pred))
    


if __name__ == '__main__':

    dfp_train = pd.read_csv('pre_train.csv', sep=',')
    dfp_test = pd.read_csv('pre_test.csv', sep=',')

    dfp_test['2'].plot.hist()

    y_train = dfp_train['2'].values
    y_test = dfp_test['2'].values

    Xp_train = select_features(dfp_train)
    Xp_test = select_features(dfp_test)

    run_model(y_train=y_train,y_test=y_test, mt='bl')
    run_model(Xp_train, y_train,Xp_test, y_test, 'lr')
    run_model(Xp_train, y_train, Xp_test, y_test, 'dtr', {"max_depth": 2, "random_state": 0})
    run_model(Xp_train, y_train, Xp_test, y_test, 'gbr', {"max_depth" : 2 , "n_estimators" : 10, 
                                                          "learning_rate" : 0.07 , "random_state" : 0,
                                                          "loss" : 'huber'})
