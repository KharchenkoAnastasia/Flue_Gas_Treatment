from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def rem_blank(df):
    df.dropna(how='all',subset=[2,3,8,9,11,12,13], inplace=True)
    df.dropna(how='all',subset=[2,3], inplace=True)
    df.dropna(how='all',subset=[11,12,13], inplace=True)
    df.dropna(how='all',subset=[3,11], inplace=True)
    df.dropna(how='all',subset=[11], inplace=True)
    df[13].fillna(0,inplace=True)
        
    for i in df.index:
        if type(df.loc[i,12])==str:
             df.drop(index=df.loc[i].name,axis =0,inplace=True)
                
def fit_missimg_val(df):
    
    imp_median = SimpleImputer( missing_values=np.nan,strategy='median')
    imp_median.fit(df)
    df=imp_median.transform(df)
    return df

    
    
def rem_duplicate(df):
    
    i=0 
    while i<len(df):
        if df[i,2]==df[i,3]==df[i,4]==0:
            df=np.delete(df,i,0)
            i-=1
        i+=1
        
    # Delete duplicates 
    df=np.unique(df, axis=0)
    return (df)

        
def rem_outliers(df):    
    i=0
    while i<6:   
         # IQR
         Q1 = np.percentile(df[:,i], 25, interpolation = 'midpoint')
         Q3 = np.percentile(df[:,i], 75, interpolation = 'midpoint')
         IQR = Q3 - Q1
         # Upper bound
         upper = np.where(df[:,i] >= (Q3+1.5*IQR))
         # Lower bound
         df=np.delete(df,upper,0)
         lower = np.where(df[:,i] <= (Q1-1.5*IQR))
         df=np.delete(df,lower,0)
         i+=1

    return (df)
 

#class normalization():
#    
#    def normalization(self,array):
#            
#        self.scaler = MinMaxScaler()
#        self.scaler.fit(array)
#        array=self.scaler.transform(array)
#        return (array)
#
#    def invert_normalize(self, xnew):
#        xnew=self.scaler.inverse_transform(xnew)
#        print(xnew)
#        
        
    