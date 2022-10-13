import numpy as np
import pandas as pd
import tensorflow.keras as K
#INDEX=0
class SlidingWindow(K.utils.Sequence):
    def __init__(self, X_df, Y_df, X_indexes,Y_indexes, batch_size, feature_inSteps=23, outSteps=4,
                features2use='all', TargetVariables='all', split_features=True ):
        if features2use=='all':
            self.FIDX=np.arange(X_df.shape[1])
        else:
            self.FIDX=[X_df.columns.get_loc(colname) for colname in features2use]
        if TargetVariables=='all':
            self.TIDX=np.arange(Y_df.shape[1])
        else:
            self.TIDX=[Y_df.columns.get_loc(colname) for colname in TargetVariables]
        self.dim = (feature_inSteps, len(self.FIDX))
        self.batch_size=batch_size
        self.X_df=X_df
        self.Y_df=Y_df
        self.X_indexes=X_indexes
        self.Y_indexes=Y_indexes
        self.split_features=split_features
        self.on_epoch_end()
        self.outSteps=outSteps
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.Y_indexes)-self.outSteps+1)/ self.batch_size))
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.X_indexes=self.X_indexes
        self.Y_indexes=self.Y_indexes
    def __data_generation(self, X_dfSLICE,Y_dfSLICE):
        'Generates data containing batch_size samples' # X : (n_samples, *dim) #y : (n_samples, outSteps)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.outSteps, len(self.TIDX)))

        # Generate data
        for i in range(self.batch_size):
            # Store input sample
            #print(X[i,:].shape, X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX].shape)
            #print(y[i,:].shape, Y_dfSLICE.iloc[i:i+self.outSteps,self.TIDX].shape)
            #print(X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX].shape, i)
            X[i,:] = X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX]
            # Store reference
            y[i,:] =  Y_dfSLICE.iloc[i:i+self.outSteps,self.TIDX]

        return X, y
    def __getitem__(self, index):
        'Generate one batch of data'
        #print(index)
        # Generate indexes of the batch
        BatchIDX_X =self.X_indexes[index*self.batch_size:(index+1)*self.batch_size + self.dim[0]-1]
        BatchIDX_Y =self.Y_indexes[index*self.batch_size:(index+1)*self.batch_size + self.outSteps-1]
        # Generate data
        X, yBatch = self.__data_generation(self.X_df.iloc[BatchIDX_X,:],self.Y_df.iloc[BatchIDX_Y,:])
        if self.dim[len(self.dim)-1]>1 and self.split_features: #Multiple inputs case
            XBatch=[]
            for feature in range(self.dim[len(self.dim)-1]):
                   XBatch.append(X[:,:,feature])
        else:
            XBatch=X
        return XBatch, yBatch

class SlidingWindowManySlices(K.utils.Sequence):
    def __init__(self, X_df, Y_df, X_list,Y_list, batch_size, feature_inSteps=23, outSteps=4,
                features2use='all', TargetVariables='all', split_features=True ):
        if features2use=='all':
            self.FIDX=np.arange(X_df.shape[1])
        else:
            self.FIDX=[X_df.columns.get_loc(colname) for colname in features2use]
        if TargetVariables=='all':
            self.TIDX=np.arange(Y_df.shape[1])
        else:
            self.TIDX=[Y_df.columns.get_loc(colname) for colname in TargetVariables]
        self.dim = (feature_inSteps, len(self.FIDX))
        self.outSteps=outSteps
        self.batch_size=batch_size
        self.X_df=X_df
        self.Y_df=Y_df
        self.cumulative_batch_count=[]
        stacksum=0
        for YID in Y_list:
            stacksum+=self.count_batches(YID)
            self.cumulative_batch_count.append(stacksum)
        self.Y_indexes=np.concatenate([Yndexes[:(self.count_batches(Yndexes)*self.batch_size)+
                                                 self.outSteps-1] for Yndexes in Y_list]) 
        self.X_indexes=np.concatenate([X_list[i][:(self.count_batches(Y_list[i])*self.batch_size)+
                                                 self.dim[0]-1] for i in range(len(X_list))]) 
        self.Y_offsets=[0] if len(Y_list)==1 else [0] + [ (self.outSteps -1)*(i+1) for i in range(len(Y_list)-1)]
        self.X_offsets=[0] if len(X_list)==1 else [0] + [ (self.dim[0] -1)*(i+1) for i in range(len(X_list)-1)]
        self.split_features=split_features
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.Y_indexes)-sum(self.Y_offsets) )/ self.batch_size))
    def count_batches(self,Yndexes):
        return int(np.floor((len(Yndexes)-self.outSteps+1)/ self.batch_size))
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.X_indexes=self.X_indexes
        self.Y_indexes=self.Y_indexes
    def __data_generation(self, X_dfSLICE,Y_dfSLICE):
        'Generates data containing batch_size samples' # X : (n_samples, *dim) #y : (n_samples, outSteps)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.outSteps, len(self.TIDX)))

        # Generate data
        for i in range(self.batch_size):
            # Store input sample
            #print(X[i,:].shape, X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX].shape)
            #print(y[i,:].shape, Y_dfSLICE.iloc[i:i+self.outSteps,self.TIDX].shape)
            #print(X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX].shape, i)
            X[i,:] = X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX]
            # Store reference
            y[i,:] =  Y_dfSLICE.iloc[i:i+self.outSteps,self.TIDX]

        return X, y
    def __getitem__(self, index):
        'Generate one batch of data'
        J=0
        for element in self.cumulative_batch_count:
            if index>= element:
                J+=1       
        #print(index)
        # Generate indexes of the batch
        BatchIDX_X =self.X_indexes[(index*self.batch_size)+self.X_offsets[J]:(index+1)*self.batch_size +self.X_offsets[J]+ self.dim[0]-1]
        BatchIDX_Y =self.Y_indexes[(index*self.batch_size)+self.Y_offsets[J]:(index+1)*self.batch_size +self.Y_offsets[J]+ self.outSteps-1]
        # Generate data
        X, yBatch = self.__data_generation(self.X_df.iloc[BatchIDX_X,:],self.Y_df.iloc[BatchIDX_Y,:])
        if self.dim[len(self.dim)-1]>1 and self.split_features: #Multiple inputs case
            XBatch=[]
            for feature in range(self.dim[len(self.dim)-1]):
                   XBatch.append(X[:,:,feature])
        else:
            XBatch=X
        return XBatch, yBatch

#class SlidingWindowIDXList(K.utils.Sequence):
#    def __init__(self, X_df, Y_df, XidList,YidList, batch_size, feature_inSteps=23, outSteps=4,
#                features2use='all', TargetVariables='all' ):
#        if features2use=='all':
#            self.FIDX=np.arange(X_df.shape[1])
#        else:
#            self.FIDX=[X_df.columns.get_loc(colname) for colname in features2use]
#        if TargetVariables=='all':
#            self.TIDX=np.arange(Y_df.shape[1])
#        else:
#            self.TIDX=[Y_df.columns.get_loc(colname) for colname in TargetVariables]
#        self.dim = (feature_inSteps, len(self.FIDX))
#        self.batch_size=batch_size
#        self.outSteps=outSteps
#        self.X_df=X_df
#        self.Y_df=Y_df
#        self.XidList=XidList
#        self.YidList=YidList
#        self.on_epoch_end()
#    def __len__(self):
#        global INDEX
#        'Denotes the number of batches per epoch'
#        return int(np.floor((len(self.YidList[INDEX])-self.outSteps+1)/self.batch_size)) 
#    def on_epoch_end(self):
#        global INDEX
#        'Updates indexes after each epoch'
#        self.X_indexes=self.XidList[INDEX]
#        self.Y_indexes=self.YidList[INDEX]
#    def __data_generation(self, X_dfSLICE,Y_dfSLICE):
#        'Generates data containing batch_size samples' # X : (n_samples, *dim) #y : (n_samples, outSteps)
#        # Initialization
#        X = np.empty((self.batch_size, *self.dim))
#        y = np.empty((self.batch_size, self.outSteps, len(self.TIDX)))
#
#        # Generate data
#        for i in range(self.batch_size):
#            # Store input sample
#            #print(X[i,:].shape, X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX].shape)
#            #print(y[i,:].shape, Y_dfSLICE.iloc[i:i+self.outSteps,self.TIDX].shape)
#            #print(X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX].shape, i)
#            X[i,:] = X_dfSLICE.iloc[i:i+self.dim[0],self.FIDX]
#            # Store reference
#            y[i,:] =  Y_dfSLICE.iloc[i:i+self.outSteps,self.TIDX]
#
#        return X, y
#    def __getitem__(self, index):
#        'Generate one batch of data'
#        BatchIDX_X =self.X_indexes[index*self.batch_size:(index+1)*self.batch_size + self.dim[0]-1]
#        BatchIDX_Y =self.Y_indexes[index*self.batch_size:(index+1)*self.batch_size + self.outSteps-1]
#        # Generate data
#        X, yBatch = self.__data_generation(self.X_df.iloc[BatchIDX_X,:],self.Y_df.iloc[BatchIDX_Y,:])
#        if self.dim[len(self.dim)-1]>1: #Multiple inputs case
#            XBatch=[]
#            for feature in range(self.dim[len(self.dim)-1]):
#                   XBatch.append(X[:,:,feature])
#        else:
#            XBatch=X
#        return XBatch, yBatch
#
#
##class INDEX_CHANGE_CALLBACK(K.callbacks.Callback):
##    def __init__(self, RANGE):
##        self.RANGE=RANGE
##    def on_epoch_end(self, epoch, logs=None):
##        rand_int=np.random.randint(0,self.RANGE)
##        global INDEXu
##        INDEX= rand_int
##        print(INDEX)
