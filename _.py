from multiprocessing import Pool

def test2(x):
    return x**2

po = Pool()

sample = [1,2,3,4,5,6,7]

if __name__=='__main__':
    res = po.map(test2, sample)
    print (res)
