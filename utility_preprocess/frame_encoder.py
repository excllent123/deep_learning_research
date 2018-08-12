from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

class PandasBaseAgent():
    def __init__(self):
        pass

    def fit(self, df):
        '''
        this shold be over-writ
        '''
        pass

    def transform(self, df):
        if not self.cols: raise('need fit first')
        r_df = df.copy()
        for col in self.cols: 
            r_df[col] = self.le[col].transform(r_df[col])
        return r_df 

    def inverse_transform(self, df):
        if not self.cols: raise('need fit first')
        r_df = df.copy()
        for col in self.cols: 
            r_df[col] = self.le[col].inverse_transform(r_df[col])
        return r_df

class PandasLabelEncoder(PandasBaseAgent):
    def __init__(self):
        super(PandasLabelEncoder, self).__init__()
    
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i]=='object']
        self.le = {}
        for col in self.cols:
            self.le[col] = LabelEncoder()
            self.le[col].fit(list(df[col]))



class PandasOneHotEncoder(PandasBaseAgent):
    def __init__(self):
        super(PandasOneHotEncoder, self).__init__()
        
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i] in ['int8', 'int32', 'int64'] ]
        self.le = {}
        for col in self.cols:
            self.le[col] = OneHotEncoder()
            self.le[col].fit(list(df[col]))

class PandasMinMaxScaler(PandasBaseAgent):
    def __init__(self):
        super(PandasMinMaxScaler, self).__init__()
        
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i] in ['int8', 
                            'int32', 'int64', 'float64', 'float32'] ]
        self.le = {}
        for col in self.cols:
            self.le[col] = MinMaxScaler()
            self.le[col].fit(df[col].values.flatten())


class PandasStandardScaler(PandasBaseAgent):
    def __init__(self):
        super(PandasStandardScaler, self).__init__()
        
    def fit(self, df):
        self.cols =  [i for i in df.columns if df.dtypes[i] in ['int8', 
                            'int32', 'int64', 'float64', 'float32'] ]
        self.le = {}
        for col in self.cols:
            self.le[col] = StandardScaler()
            self.le[col].fit(df[col].values.flatten())
