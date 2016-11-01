
from __future__ import division
import numpy as np
#from keras.engine import Layer
# =============================================================================
# encode : (Ground Truth Box | Image ) -> Ground Truth Y
# decode : Predict Tensor Y ->
# the encode and decode are symmetric in logic, however since the prediction
# may contain multi-objects, the output object should be more complicated
# =============================================================================


class YoloDetect(object):
    def __init__(self, numClass=20, rawImgW=448, rawImgH=448, S=7, B=2, thred=0.2):
        self.S = S
        self.B = B
        self.C = numClass
        self.W = rawImgW
        self.H = rawImgH
        self.threshold = thred
        self.iou_threshold=0.5
        self.classMap  =  ["aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    def set_class_map(self, mappingList):
        assert type(mappingList)==list
        assert len(mappingList) == self.C
        self.classMap=mappingList

    def encode(self, classid, cX, cY, boxW, boxH):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        assert int(classid) <= int(C-1)

        classProb  = np.zeros([S, S, C   ])
        confidence = np.zeros([S, S, B   ])
        boxes      = np.zeros([S, S, B, 4])

        # target the center grid
        gridX, gridY = W/S, H/S
        tarIdX, tarIdY = int(cX/gridX) , int(cY/gridY)

        # assign the true value
        classProb[tarIdX, tarIdY, classid] = 1.0
        confidence[tarIdX, tarIdY, :      ] = 1.0

        # x,y,w,h
        boxes[tarIdX, tarIdY, :, 0] = (cX/gridX) - int(cX/gridX)
        boxes[tarIdX, tarIdY, :, 1] = (cY/gridY) - int(cY/gridY)
        boxes[tarIdX, tarIdY, :, 2] = np.sqrt(boxW/W)
        boxes[tarIdX, tarIdY, :, 3] = np.sqrt(boxH/H)

        return np.concatenate([classProb.flatten(),confidence.flatten(),
                               boxes.flatten()])

    def decodeT(self, prediction):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        classProb  = np.reshape(prediction[0:S*S*C], (S,S,C))
        confidence = np.reshape(prediction[S*S*C: S*S*(C+B)], (S,S,B)) #
        boxes      = np.reshape(prediction[S*S*(C+B):],(S,S,B,4))
        return classProb, confidence, boxes

    def interpret(self, prediction,threshold=0.2 ,only_objectness=0):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        classProb ,confidence, boxes = self.decodeT(prediction)
        # offset (7,7,2) mask, retrieve from offset
        offset = np.transpose(np.reshape(np.array([np.arange(S)]*S*B),(B,S,S)),(1,2,0))
        boxes[:,:,:,1] += offset
        boxes[:,:,:,0] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / float(S)

        # retrieve from sqrt
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

        # retrieve from normalization
        boxes[:,:,:,0] *= self.W
        boxes[:,:,:,1] *= self.H
        boxes[:,:,:,2] *= self.W
        boxes[:,:,:,3] *= self.H

        # Pr(class|Obj) * Pr(obj) = Evaluate Proba
        eProbs = np.zeros((S,S,B,C))
        for i in range(B):
            for j in range(C):
                eProbs[:,:,i,j]=np.multiply(classProb[:,:,j],confidence[:,:,i])

        # Filter
        filter_mat_probs = np.array(eProbs >= self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)

        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = eProbs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        # select the best pridect box with the ideal similar to nms
        # if there are 2 same probs, not likely, random pick one
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold :
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classMap[classes_num_filtered[i]],boxes_filtered[i][1],boxes_filtered[i][0],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    def selective_iou(self, boxTensorTru1, boxTensorPre2):
        '''
        input B boxes Tensor
        output_w matrix that represent the 1^obj_ij
        '''
        output_weight = np.zeros([S,S,B])
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    pass
        for candidate in range(B):
            boxTensorPre2[:,:,candidate:]
        pass

    def loss(self, truY, preY):
        loss = pow((truY - preY),2)
        # chose box to penalize
        # adding penalty function
        truClaProb ,truConfi, truBoxes = self.decode(truY)
        claProb ,confi, boxes = self.decode(preY)
        COORD = .5
        NOOBJ = .1
        iouT = self.iouTensor(truBoxes,boxes)
        pass

    def boxArea(self, box):
        return box[:,:,:,2]*box[:,:,:,3]

    def iouTensor(self, box1, box2):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        assert box1.shape == box2.shape == (S,S,B,4)
        print box1[:,:,:,0].shape
        minTop = min(box1[:,:,0,0]+0.5*box1[:,:,0,2],
                     box2[:,:,0,0]+0.5*box2[:,:,0,2])

        maxBot = max(box1[:,:,:,0]-0.5*box1[:,:,:,2],
                     box2[:,:,:,0]-0.5*box2[:,:,:,2])

        minR   = min(box1[:,:,:,1]+0.5*box2[:,:,:,3],
                     box1[:,:,:,1]+0.5*box2[:,:,:,3])

        maxL   = max(box1[:,:,:,1]-0.5*box2[:,:,:,3],
                     box1[:,:,:,1]-0.5*box2[:,:,:,3])

        # intersection
        inters = (minTop-maxBot).clip(min=0)* (minR-maxL).clip(min=0)

        # IOU
        return inters/ (self.boxArea(box1)+ self.boxArea(box2)- inters)

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],
                 box2[0]+0.5*box2[2])-\
             max(box1[0]-0.5*box1[2],
                 box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],
                 box2[1]+0.5*box2[3])-\
             max(box1[1]-0.5*box1[3],
                 box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection/ (box1[2]*box1[3] + box2[2]*box2[3] -intersection)


if __name__ =='__main__':
    detector2 = YoloDetect(S=4, B=3)
    # a = detector.encode(3, 440, 440, 112, 12)
    # print a[1000:]
    # print detector.decodeT(a)
    # print detector.interpret(a)
    for cX,cY in [(50,50),(440,440)]:

        detector = YoloDetect()
        a = detector.encode(3, cX, cY, 112, 12)
        b = detector.encode(3, 52, 34,112,12)
        #b = detector2.encode(3, cX,cY,112,13)#

        #print detector.interpret(a)
        _,_,box1 = detector.decodeT(a)
        _,_,box2 = detector.decodeT(b)
        print detector.iouTensor(box1,box2)

#        #print detector.interpret_output(a)#

#        print ('====')
#        print detector2.interpret(b)
#    print ('==H======H==')
#    print a[1000:]
    #print detector.decodeT(a)


