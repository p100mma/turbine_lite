import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def is_callable(obje):
    """Check if obje is callable i.e. a function. """
    return hasattr(obje,'__call__')

def LinkParts(Array,L,R):
    """ Links slices of defined in pairs in L,R of Array along first axis (rows in 2d case)."""
    return pd.concat([Array.iloc[L[i]:R[i],:] for i in range(len(L))])
def GetSlices(L,R):
    """ Returns a np.array of indexes that can be passed to pd.Dataframe .iloc indexing among first axis
        to get the result as above."""
    return np.concatenate([[i for i in range(L[j],R[j])] for j in range(len(L))])

def TransformCols(X_series,feature_mapping,L,R):
    """ Apply various transformations on columns on all samples in the same way.
        feature_mapping should be a dictionary, where key is a name of a column and a value a spec of transformation.
        specifications:
        -('interpolate', named_args_dict) - call .interpolate() method with **named_args on X_series[key].
        -(column_function, [name1,name2...], OTHER_ARGS ) - X_series[key]=column_function(*ARGS) 
                                              where ARGS=[X_series[name1],X_series[name2]...]+OTHER_ARGS"""
    X_tformed=X_series.copy()
    for colname in feature_mapping.keys():
        if is_callable(feature_mapping[colname][0]):
            FUN, arg_cols, OTHER_ARGS=feature_mapping[colname]
            for j in range(len(L)):
                ARGS= [X_series.iloc[GetSlices([L[j]],[R[j]]),X_series.columns.get_loc(argcol)] for argcol in arg_cols]
                ARGS=ARGS+OTHER_ARGS
                X_tformed.iloc[GetSlices([L[j]],[R[j]]),X_series.columns.get_loc(colname)]= FUN(*ARGS)   
        elif feature_mapping[colname][0]=='interpolate':
            for j in range(len(L)):
                ToFill=X_tformed.iloc[GetSlices([L[j]],[R[j]]),X_series.columns.get_loc(colname)]
                X_tformed.iloc[GetSlices([L[j]],[R[j]]),X_series.columns.get_loc(colname)]=ToFill.interpolate(**feature_mapping[colname][1])
        else:
            print("No match for specification with ",colname," key, specification[0] is ",feature_mapping[colname][0])
    return X_tformed

def EstimatorCols(X_series,feature_names, Ldict,Rdict, Estimator_callable):
    """ fit sklearn Estimator on training series and transform on all samples.
        returns data frame (X_series) but with columns with names in feature_names transformed
        by Estimator."""
    X_scaled=X_series.copy()
    Estimators={}
    Big_L= Ldict["train"]+ Ldict["validation"]+Ldict["test"]
    Big_R= Rdict["train"]+ Rdict["validation"]+Rdict["test"]
    for feature_name in feature_names:
        Estimators[feature_name]= Estimator_callable()
        FIDX=X_series.columns.get_loc(feature_name)
        Estimators[feature_name].fit(LinkParts(X_series, Ldict["train"], Rdict["train"]).iloc[:,FIDX:FIDX+1] )
        for j in range(len(Big_L)):
            X_scaled.iloc[GetSlices([Big_L[j]],[Big_R[j]]),FIDX:FIDX+1]=Estimators[feature_name].transform(
                                                                                                X_series.iloc[GetSlices([Big_L[j]],[Big_R[j]]),FIDX:FIDX+1])
    return X_scaled, Estimators   

def AutoIntStanScale(X_series, feature_mapping, Ldict, Rdict, keep_estimators=False):
    """ feature_mapping should be a dictionary, keys are column names (variables),
         and values- lists of specifications, currently 3 available: interpolate, standardize, scale."""
    if keep_estimators:
        Estimators={}
    X_tfed=X_series.copy()
    Big_L= Ldict["train"]+ Ldict["validation"]+Ldict["test"]
    Big_R= Rdict["train"]+ Rdict["validation"]+Rdict["test"]
    intermap={}
    any_inter=False
    for colname in feature_mapping.keys():
        SpecList=feature_mapping[colname]
        for element in SpecList:
            if element=='interpolate':
                any_inter=True
                intermap[colname]=('interpolate',{'method':'linear','limit_direction':'both'})
    if any_inter:
        X_tfed=TransformCols(X_tfed,intermap, Big_L, Big_R)
    stList=[]
    any_st=False
    for colname in feature_mapping.keys():
        SpecList=feature_mapping[colname]
        for element in SpecList:
            if element=='standardize':
                any_st=True
                stList.append(colname)
    if any_st:
        X_tfed, Stands=EstimatorCols(X_tfed, stList, Ldict, Rdict, StandardScaler)
        if keep_estimators:
            Estimators['standardize']=Stands              
    scList=[]
    any_sc=False
    for colname in feature_mapping.keys():
        SpecList=feature_mapping[colname]
        for element in SpecList:
            if element=='scale':
                any_sc=True
                scList.append(colname)
    if any_sc:
        X_tfed, Scals=EstimatorCols(X_tfed, scList, Ldict, Rdict, MinMaxScaler)
        if keep_estimators:
            Estimators['scale']=Scals
    if keep_estimators:
        return X_tfed, Estimators
    else:
        return X_tfed              
