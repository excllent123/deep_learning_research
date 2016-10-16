# import the necessary packages
import common_func
import numpy as np
import gc
import cv2
# http://stackoverflow.com/questions/22440421/python-is-the-garbage-collector-run-before-a-memoryerror-is-raised
# if there are hidden reference cycles
# use gc.collect() to manually release the memory in each iteration


def singleton(class_):
  instances = {}
  def getinstance(*args, **kwargs):
    if class_ not in instances:
        instances[class_] = class_(*args, **kwargs)
    return instances[class_]
  return getinstance

@singleton
class Memo(object):
    def __init__(self, numLabel):
        self.boxes={}
        self.prob ={}
        for i in range(numLabel):
            self.boxes[i]=[]
            self.prob[i]=[]

    def addNode(self, labelID, box, prob):
        assert type(box)==tuple
        self.boxes[labelID].append(box)
        self.prob[labelID].append(prob)

    def extratResult(self,labelID):
        try :
            return self.boxes[labelID], self.prob[labelID]
        except Exception as err:
            print (err)

# windim = (x,y) = 32, 32 this case
def detect(image, model, winDim,
                   pyScale=20,winStep=20,
                   minProb=0.9995, numLabel=2, negLabel=[0]):
    '''
    numLable = y-label number
    in this case, we have 2 lebels 0 and 1
    negLable is the background, or any object lable that we dont detect
    '''
    assert type(winDim)==tuple
    height, width, channel = image.shape

    memo = Memo(numLabel)
    orig = image.copy()

    # loop over the image pyramid
    for layer in common_func.pyramid(image, scale=pyScale, minSize=winDim):
        # determine the current scale of the pyramid
        scale = image.shape[0] / float(layer.shape[0])

        # loop over the sliding windows for the current pyramid layer
        for (x, y, wD) in common_func.sliding_window(layer, winStep, winDim):
            # grab the dimensions of the window
            (winH, winW) = wD.shape[:2]
            # ensure the window dimensions match the supplied sliding window dimensions
            if winH == winDim[1] and winW == winDim[0]:
                # reshape
                wD = wD.reshape(1,channel,winW,winH)

                #return window [0][0]
                probS = model.predict_proba(wD,verbose=0)[0]

                # check to see if the classifier has found an object with sufficient
                # probability
                for label, prob in enumerate(probS):
                    if prob>minProb:
                        (startX, startY) = (int(scale * x), int(scale * y))
                        endX = int(startX+(scale * winW))
                        endY = int(startY+(scale * winH))
                        #
                        box = (startX, startY, endX, endY)
                        memo.addNode(label, box, prob)
    print ('END Detect')
    for label in range(numLabel):
        if label in negLabel:
            continue
        boxes, probs = memo.extratResult(label)
        pick = common_func.non_max_suppression(np.array(boxes), probs, 0.6)
        # loop over the allowed bounding boxes and draw them
        for (startX, startY, endX, endY) in pick:
            if startY < 10 or startX < 10 or abs(startX - endX)>50 or abs(startX - endX)<15 :
                pass
            else:
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, label*55), 2)
    return orig
