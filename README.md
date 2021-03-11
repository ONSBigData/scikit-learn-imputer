# Scikit learn imputer

![](/download.png)

Use SklearnImputer to impute mixed data sets using scikit-learn algorithms.

The tool has some great features:
- it can impute both catgeorical (using classifiers) and continuous (using regression) data. The user only needs to define what variables are catgeorical. One-hot encoding is done internally.
- it will impute all the missing values in the data set (it does do this one at a time i.e. univariate).
- it performs a simulation study - training the algorithm on 90% (by default) of the non-missing data and testing on 10% (by default) of the non-missing data. This allows you to assess performance.
- trained models are saved automatically and then can re-loaded at a later stage - so if your model takes a long time to train, you only to train it once. This setting can also be switched off, if you want to do the transformation each time.
- saved models can be used to impute different data sets with the same features. You can use the validate method to assess how appropriate the saved models are.
- you can use the select_model method to help decide what classifier and regressor you should use - based on performance (accuarcy / mean squared error, training time and deployment time)
- you can make bespoke classifiers / regressors and input these (e.g. I made HierarchicalHotDeck).

## Example Usage 

Example usage for SklearnImputer

For a simple imputation problem where you do not want to save a trained model. Your data set df should have
nan representing missing values.
```
>>> from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
>>> from SklearnImputer import SklearnImputer
>>> from pandas import DataFrame
>>> from numpy import nan
>>>
>>> df = DataFrame({'A': [1, nan, 2, 3, 4], 'B': [1, 3, 4, 3, 4], 'C': [nan, nan, 2, 1, 2]})
>>>
>>> imputer = SklearnImputer(input_data=df)
>>> output = imputer.fit(classification=DecisionTreeClassifier(), regression=DecisionTreeRegressor(), transform=True)
>>>
>>> output['imputed_data']
```
        A   B   C
    0  1.0  1  1.0
    1  4.0  3  1.0
    2  2.0  4  2.0
    3  3.0  3  1.0
    4  4.0  4  2.0
    
If you want to use the tool for a mix of categorical and continuous variables:

```
>>> df = DataFrame({'A': [1, nan, 2, 3, 4], 'B': [1, 3, 4, 3, 4], 'C': [nan, nan, 2, 1, 2],
>>>                 'D': [1, 0, nan, 1, 0], 'E': [1, 1, nan, 1, nan], 'F': [0, nan, nan, 1, 0]})
>>>
>>> imputer = SklearnImputer(input_data=df, categorical=['D', 'E', 'F'])
>>> output = imputer.fit(classification=DecisionTreeClassifier(), regression=DecisionTreeRegressor(), transform=True)
>>>
>>> output['imputed_data']
```

       A    B    C    D    E    F
    0  1.0  1  1.0  1.0  1.0  0.0
    1  4.0  3  1.0  0.0  1.0  0.0
    2  2.0  4  2.0  1.0  1.0  0.0
    3  3.0  3  1.0  1.0  1.0  1.0
    4  4.0  4  2.0  0.0  1.0  0.0
    

If you want to save the model to apply it at a later stage or to avoid re-training the model when reproducing results:

```
>>> df = DataFrame({'A': [nan, 4, 2, 3, 4], 'B': [nan, 3, 4, 3, 4], 'C': [1, nan, 2, 1, 2],
>>>                           'D': [1, 0, nan, nan, 0], 'E': [1, nan, 1, 1, 0], 'F': [0, 1, 0, nan, nan]})
>>>
>>> imputer = SklearnImputer(input_data=df, categorical=['D', 'E', 'F'], save_models_to=r'./saved_model.z')
>>> output = imputer.fit(classification=DecisionTreeClassifier(), regression=DecisionTreeRegressor(), transform=False)
>>>
>>> model_applied_output = imputer.transform()
>>> model_applied_output['imputed_data']
```

         A    B    C    D    E    F
    0  3.0  3.0  1.0  1.0  1.0  0.0
    1  4.0  3.0  1.0  0.0  1.0  1.0
    2  2.0  4.0  2.0  0.0  1.0  0.0
    3  3.0  3.0  1.0  0.0  1.0  1.0
    4  4.0  4.0  2.0  0.0  0.0  0.0
