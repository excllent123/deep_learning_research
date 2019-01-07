def get_gain_by_row(row, oes_feats):
    return row[oes_feats].values.mean()

def auto_select_z_peak(df):
    wcols = [str(i) for i in np.arange(200, 870.5, 0.5)]
    zwcols = [i for i in wcols if float(i) > 400 and float(i) < 850]
    zwcols = df[zwcols].median()
    zwcols = zwcols[zwcols < 10000].index
    return zwcols

def z_alignment(df, oes_feats):
    zwcols = auto_select_z_peak(df)
    df['v_oes_gain'] = [ get_gain_by_row(df.iloc[i], oes_feats=zwcols) 
                        for i in range(len(df))]
    
    train = df[df['train_test']=='train']
    train_bsl = train['v_oes_gain'].median()#/float(len(oes_feats))
    #print(train_bsl)

    test = df[df['train_test']=='test']
    
    train['v_oes_gain'] = train['v_oes_gain'].apply(lambda x : train_bsl/float(x))
    test['v_oes_gain'] = test['v_oes_gain'].apply(lambda x : train_bsl/float(x))

    for oes_col in oes_feats:
        train[oes_col] = train[oes_col]*train['v_oes_gain']
        test[oes_col] = test[oes_col]*test['v_oes_gain']
    #print(train['v_oes_gain'])
    return train, test
    
def z_alignment_by_ch(df, oes_feats):
    res_train = []
    res_test = []
    zwcols = auto_select_z_peak(df)
    df['v_oes_gain'] = [ get_gain_by_row(df.iloc[i], oes_feats=zwcols) 
                        for i in range(len(df))]
    
    for i in list(set(df['tool_ch'])):
        temp_df = df[df['tool_ch']==i]
        
        train = temp_df[temp_df['train_test']=='train']
        test = temp_df[temp_df['train_test']=='test']
        
        train_bsl = train['v_oes_gain'].median()#/float(len(oes_feats))

        train['v_oes_gain'] = train['v_oes_gain'].apply(lambda x : train_bsl/float(x))
        test['v_oes_gain'] = test['v_oes_gain'].apply(lambda x : train_bsl/float(x))

        for oes_col in oes_feats:
            train[oes_col] = train[oes_col]*train['v_oes_gain']
            test[oes_col] = test[oes_col]*test['v_oes_gain']
            
        res_train.append(train)
        res_test.append(test)
    train = pd.concat(res_train)
    test  = pd.concat(res_test)
    return train, test
