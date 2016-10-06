# import the necessary packages
import common_func

import gc
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
def detection_test(x, model, winDim, pyramidScale=20,winStep=20, minProb=0.9995, numLabel=2):
    '''
    numLable = y-label number
    in this case, we have 2 lebels 0 and 1
    '''
    assert type(x)==int
    assert x>=0
    image = vid.get_data(x)
    image = imutils.resize(image, width=200)

    memo = Memo(numLabel)

    orig = image.copy()
    img_gray = cv2.cvtColor( image , cv2.COLOR_BGR2GRAY)
    # loop over the image pyramid
    for layer in common_func.pyramid(img_gray, scale=pyramidScale, minSize=winDim):
        # determine the current scale of the pyramid
        scale = image.shape[0] / float(layer.shape[0])

        # loop over the sliding windows for the current pyramid layer
        for (x, y, window) in common_func.sliding_window(layer, winStep, winDim):
            # grab the dimensions of the window
            (winH, winW) = window.shape[:2]


            # ensure the window dimensions match the supplied sliding window dimensions
            if winH == winDim[1] and winW == winDim[0]:
                window = window.reshape(1,1,winW,winH)

                #return window [0][0]
                probS = model.predict_proba(window,verbose=0)[0]

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
        boxes, probs = memo.extratResult(label)
        pick = common_func.non_max_suppression(np.array(boxes), probs, 0.7)
        # loop over the allowed bounding boxes and draw them
        for (startX, startY, endX, endY) in pick:
            if startY < 10 or startX < 10 or abs(startX - endX)>50 or abs(startX - endX)<15 :
                pass
            else:
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, label*255), 2)
    return orig
