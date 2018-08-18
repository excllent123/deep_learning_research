from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

class PandasBaseAgent():
    def __init__(self, except_cols=[]):
        if type(except_cols)== str:
            except_cols = [except_cols]
        self.except_cols= except_cols

    def fit(self, df):
        '''
        this shold be over-writ
        '''
        pass

    def transform(self, df):
        if not self.cols: raise('need fit first')
        r_df = df.copy()
        for col in self.cols: 
            try:
                r_df[col] = self.le[col].transform(r_df[col].values.reshape(-1, 1))
            except Exception as e : 
                print(col, ' : ' ,e)
        return r_df 

    def inverse_transform(self, df):
        if not self.cols: raise('need fit first')
        r_df = df.copy()
        for col in self.cols: 
            r_df[col] = self.le[col].inverse_transform(r_df[col].values.reshape(-1, 1))
        return r_df

class PandasLabelEncoder(PandasBaseAgent):
    def __init__(self, *arg, **kwargs):
        super(PandasLabelEncoder, self).__init__(*arg, **kwargs)
    
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i]=='object' ]
        self.cols = [i for i in self.cols if i not in self.except_cols]
        self.le = {}
        for col in self.cols:
            self.le[col] = LabelEncoder()
            self.le[col].fit(df[col].values.reshape(-1, 1))



class PandasOneHotEncoder(PandasBaseAgent):
    def __init__(self, *arg, **kwargs):
        super(PandasOneHotEncoder, self).__init__(*arg, **kwargs)
        
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i] in ['int8', 'int32', 'int64'] ]
        self.cols = [i for i in self.cols if i not in self.except_cols]
        self.le = {}
        for col in self.cols:
            self.le[col] = OneHotEncoder()
            self.le[col].fit(df[col].values.reshape(-1, 1))

class PandasMinMaxScaler(PandasBaseAgent):
    def __init__(self, *arg, **kwargs):
        super(PandasMinMaxScaler, self).__init__(*arg, **kwargs)
        
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i] in ['int8', 
                            'int32', 'int64', 'float64', 'float32'] ]
        self.cols = [i for i in self.cols if i not in self.except_cols]
        self.le = {}
        for col in self.cols:
            self.le[col] = MinMaxScaler()
            self.le[col].fit(df[col].values.reshape(-1, 1))


class PandasStandardScaler(PandasBaseAgent):
    def __init__(self, *arg, **kwargs):
        super(PandasStandardScaler, self).__init__(*arg, **kwargs)
        
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i] in ['int8', 
                            'int32', 'int64', 'float64', 'float32'] ]
        self.cols = [i for i in self.cols if i not in self.except_cols]
        self.le = {}
        for col in self.cols:
            self.le[col] = StandardScaler()
            self.le[col].fit(df[col].values.reshape(-1, 1))
