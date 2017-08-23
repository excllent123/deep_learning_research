

tax_china = [(   4800  , 0    ),
       (   1500  , 0.03 ),
       (   3000  , 0.1  ), 
       (   4500  , 0.2  ),
       (   26000 , 0.25 ),
       (   20000 , 0.3  ),
       (   25000 , 0.35 ),
       ( 9999999 , 0.45 )]



def count_tax(x, tax_rule):
    res=0
    for i, j in tax:
        if x-i>0:
            x-=i
            res+=i*j
        else:
            res+=x*j
            return res
