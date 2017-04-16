'''
this is a module to paser the json file
'''

import json

class JsonPaser:
  def __init__(self, confPath):
    # load and store the configuration and update the object's dictionary
    conf = json.loads(open(confPath).read())
    self.__dict__.update(conf)


class ConfPaser:
  def __init__(self, config):
    pass
 
class RUNNER:
  def __init__(self):
    pass 
  def run():
    os.popen('python yolo_train.py "%s"'%toolObj).read()