

def singon(cls):
    instan = {}

    def check (*args,**kwargs):
        if cls not in instan:
            instan[cls]=cls(*args,**kwargs)
        return instan[cls]
    return check

@singon
class test():
    def run(self,x):
        print x

a = test()

b = test()

assert id(a)==id(b)
