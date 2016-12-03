import commentjson as json


class ConFlag:
    def __init__(self, confPath):
        conf = json.loads(open(confPath).read())
        self.__dict__.update(conf)

    def __getitem__(self, key):
        return self.__dict__.get(key,None)


