import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

import seaborn as sns
sns.set(font_scale=2)
class eda():
    """ This class is just for some basic analysis 
    By : Mathias Godwin & Nathaniel Guimond
    """

    # filling Na with the mode
    def fillNa(dataframe=None, X_train=None, y_train=None,
               by_mean=False, by_mode=False, inplace=False):
        """ This function typically fill the NA values with the mode in a given feature """
#         for cols in dataframe:
#         if impute:
#             from sklearn.impute import SimpleImputer
#             from sklearn.preprocessing import LabelEncoder
#             
#             encoder = LabelEncoder()
#             X_train = encoder.fit_transform(X_train)
#             y_train = encoder.transform(y_train)
#             
#             imputer = SimpleImputer(strategy='most_frequent',fill_value='mode', copy=True)
#             imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
#             imputed_y_train = pd.DataFrame(imputer.transform(y_train))
                
                # replacing column
#             imputed_X_train.columns = X_train.columns
#             imputed_y_train.columns = y_train.columns

#             a = missing_by_percount(dataframe)
#             b = pd.DataFrame([dataframe.drop([colname], axis=1, inplace=True) 
#              for colname in a['Column Name'] 
#              if a['Percentage Missing'].values.any() >= 50])
#                 if droplarge:
#                     _getterFunction = pd.DataFrame(missing_by_percount(dataframe))
#                     _getValue = [rowname for rowname in _getterFunction.loc[:,'Column Name'].values.any() 
#                              if _getterFunction.loc[:, 'Percentage Missing'].values.any().item() 
#                                                     >= 50] 
#                 dataframe.drop(_getValue, axis=1, inplace=True)
                #_getterFunction.loc[:, 'Percentage Missing'] > 50
#                 dataframe.drop([_getValue == True],axis=1, inplace=True)
                
#                 for i in _getValue:
#                     pass
#                     if True:
#                         pass
#                         dataframe.drop([i.index], inplace=True)
#                     else:
#                         pass
#                 dataframe.drop([])
#                 ahoy = dataframe.drop([])
        if by_mode:
            shoot = [colname 
                     for colname in  dataframe.columns 
                     for colname in dataframe[f'{colname}'].mode()]
            for boom in shoot:
                for colname in dataframe.columns:
                    dataframe[f'{colname}'].fillna(value=boom, inplace=inplace)
        if by_mean:
            shooter = [colname
                     for colname in dataframe.columns
                   
                     if dataframe[colname].dtype in ['int64', 'float64']
#                      for colname in dataframe[f'{colname}'].mean()
                    ]
            shoot =  [colname 
                     for colname in  shooter 
                     for colname in dataframe[f'{colname}'].mean()] 
            for boom in shoot:
                for colname in dataframe.columns:
                    dataframe[f'{colname}'].fillna(value=boom, inplace=inplace)
                    

        return dataframe
    
    # return features with missing values
    def missing_col(dataFrame):
        """ Return columns and the total missing values count """
        missingCols = (dataFrame.isnull().sum())
        return missingCols[missingCols > 0]
    
    # getting the percent of missing values for every column
    def missing_by_percount(dataFrame):
        ''' The function get the percent of missing values for every column ''' 
        values_null = dataFrame.isnull().sum().sort_values(ascending=False)
        percentage = round(values_null / len(dataFrame)*100, 2)
        total = pd.concat([values_null, percentage], axis=1,
                      keys=['Total', 'Percentage Missing']).reset_index()
        return total.rename(columns=({'index':'Column Name'}))

    def featureValue_counts(dataFrame, feature):
        """This function takes in a dataframe and a column 
           and finds the percentage of the values"""
        
        percentage = pd.DataFrame(round(dataFrame.loc[:,feature].value_counts(
                                 dropna=False, normalize=True)*100,2))
        total = pd.DataFrame(dataFrame.loc[:,feature].value_counts(dropna=False))
        total.columns = ["Total"]
        percentage.columns = ['Percentage']
        return pd.concat([total, percentage], axis = 1)
    
    # CODE UNDER CONSTRUCTION , DO NOT USE YET !
    # This function counts the values of the features in a data frame
    def featuresValue_counts(dataFrame, features):
        """ I guess this is more than useless """
        values = []
        percentage = []
        
        for value in features:
            if len(features) > 1:
                values.append(dataFrame.loc[:, value].value_counts(dropna=False))
                percentage.append(pd.DataFrame(round(
                         dataFrame.loc[:, value].value_counts(
                         dropna=False,
                         normalize=True)*100, 2)))
        
            else:
                values = dataFrame.loc[:, features].value_counts(dropna=False)
                percentage = pd.DataFrame(round(dataFrame.loc[:, features].value_counts(
                            dropna=False, normalize=True)*100, 2))
                percentage.columns = ['Percentage']
        values = pd.DataFrame(values).reset_index()
        percentage = pd.DataFrame(percentage).reset_index()
        total = pd.concat([values, percentage], axis=1)
        return 
    
    # Test for correlation in the data
   
    def corrplot(dataframe, plot=True):
        """ Gives the heatmap of the correlation between all columns """
        if plot:
            ## heatmeap to see the correlation between features. 

            mask = np.zeros_like(dataframe.corr(), dtype=np.bool)
            plt.subplots(figsize = (15,12))
            sns.heatmap(dataframe.corr(), 
            annot=True,
            mask = mask,
            cmap = 'binary', # in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='gray',
            fmt='.2g',
            center = 0,
            square=True)
            plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);
        else:
            pass
#             
            
        return 
    
    def corr(dataframe, target):
        """ returns the correlation between the values of a dataframe and a target variable """
        correlation = dataframe.corr()[target].sort_values(ascending=False)
        return correlation
    
    def drop_acolumn(dataframe, colname, axis=1, inplace=True):
        # This would be rarely used 
        """ To drop a column from the dataframe """
        return dataframe.drop(columns=colname, axis=axis, inplace=inplace)
    
    def drop_cat_columns(dataframe, axis=1, inplace=True):
    # I guess this would be rarely used
        """ It takes the dataframe and drop every categorical column """
        return dataframe.drop([colname 
                                   for colname in dataframe.columns 
                                   if dataframe[colname].dtype == 'object'], 
                                   axis=axis,
                                   inplace=inplace)
        
    def drop_bunchMissing(dataframe, na_limit=50, axis=1, inplace=True):
        """ This funtion drop every column with a given amount of allowed
            missing values which is the na_limit
            
            na_limit : takes the maximum percentage of missing value
                           and drop greater values
         """
        bunchmis = pd.DataFrame(dataframe.isnull().sum().sort_values(ascending=False)/
                       len(dataframe)*100).reset_index()
        bunchlist = bunchmis.loc[(bunchmis[0] >= na_limit), 'index'].to_list()
        dataframe.drop(bunchlist, axis=axis, inplace=inplace)
        return dataframe
    def drop_missing_cols(dataframe, axis=1, inplace=True):
        """ This function drop all columns with missing values """
        return dataframe.drop((colname
                              for colname in dataframe.columns
                              if dataframe[colname].isnull().any()), 
                              axis=axis, 
                              inplace=inplace)
    def cross_val(X_train=None, X_valid=None, y_train=None, model=None):
        # Applying k-Fold Cross Validation
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
        model_pred = model.predict(X_valid)
        def fit_and_evaluate(model):
    
    # Train the model
            model.fit(X_train, y_train)
    
    # Make predictions and evalute
        
#             model_cross =
            return  cross_val(X_train, y_train, model)
        return pd.DataFrame({'prediction':model_pred})#, accuracies.mean()
    # Return the performance me
#         return accuracies
    #     @cross_val(X_train, y_train, model)
#     @property()

#     @fit_and_evaluate(model)
  # Takes in a model, trains the model, and evaluates the model on the test set
# tric
    def mae(prediction, validator):
        return sum(abs(prediction - validator))/len(prediction)

        