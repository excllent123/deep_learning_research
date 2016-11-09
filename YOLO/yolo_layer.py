
from __future__ import division
import numpy as np
from keras.engine import Layer

# =============================================================================
# encode : (Ground Truth Box | Image ) -> Ground Truth Y
# decode : Predict Tensor Y ->
# the encode and decode are symmetric in logic
# =============================================================================


class YoloDetector(Layer):
    def __init__(self, numCla=20, rImgW=448, rImgH=448, S=7, B=2):
        self.S = S
        self.B = B
        self.C = numCla
        self.W = rImgW
        self.H = rImgH
        self.iou_threshold=0.5
        self.classMap  =  ["aeroplane", "bicycle", "bird", "boat", "bottle", 
                           "bus", "car", "cat", "chair", "cow", "diningtable",
                           "dog", "horse", "motorbike", "person", "pottedplant",
                           "sheep", "sofa", "train","tvmonitor"]

    def set_class_map(self, mappingList):
        assert type(mappingList)==list ; assert len(mappingList) == self.C
        self.classMap=mappingList

    def encode(self, annotations):
        ''' annotations : nested list contained
        '''
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.

        # init
        classProb  = np.zeros([S, S, C   ])
        confidence = np.zeros([S, S, B   ])
        boxes      = np.zeros([S, S, B, 4])

        for classid, cX, cY, boxW, boxH in annotations:
            assert int(classid) <= int(C-1)

            # Target the center grid
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

        yield np.concatenate([classProb.flatten(),confidence.flatten(),
                               boxes.flatten()])

    def traDim(self, pred, mode=3):
        ''' Dimension Transformation of Tensor '''
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H

        if mode == 3 :
            classProb  = np.reshape(pred[0:S*S*C]         , (S,S,C))
            confidence = np.reshape(pred[S*S*C: S*S*(C+B)], (S,S,B)) 
            boxes      = np.reshape(pred[S*S*(C+B):]      , (S,S,B,4))

        elif mode == 2 :
            classProb  = np.reshape(pred[0:S*S*C]         , (S*S,C))
            confidence = np.reshape(pred[S*S*C: S*S*(C+B)], (S*S,B)) 
            boxes      = np.reshape(pred[S*S*(C+B):]      , (S*S,B,4))

        return classProb, confidence, boxes        

    def decode(self, prediction,threshold=0.2 ,only_objectness=0):
        '''
        this part is modified from https://github.com/gliese581gg/YOLO_tensorflow
        '''
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        classProb ,confidence, boxes = self.traDim(prediction, mode =3)

        # offset (7,7,2) mask, retrieve from offset
        offset = np.transpose(np.reshape(np.array([np.arange(S)]*S*B),(B,S,S)),(1,2,0))
        boxes[:,:,:,1] += offset
        boxes[:,:,:,0] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / float(S)

        # retrieve from sqrt
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

        # retrieve from normalization
        boxes[:,:,:,0] *= self.W ; boxes[:,:,:,1] *= self.H
        boxes[:,:,:,2] *= self.W ; boxes[:,:,:,3] *= self.H

        # Pr(class|Obj) * Pr(obj) = Evaluate Proba
        eProbs = np.zeros((S,S,B,C))
        for i in range(B):
            for j in range(C):
                eProbs[:,:,i,j]=np.multiply(classProb[:,:,j],confidence[:,:,i])

        # Filter
        filter_mat_probs = np.array(eProbs >= threshold,dtype='bool')
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

    def loss(self, truY, preY, COORD=5. , NOOBJ=.5 , loss_=0):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H

        truCP ,truConf, truB = self.traDim(truY, mode=2)
        preCP ,preConf, preB = self.traDim(preY, mode=2)

        # Select for responsible box which with max IOU
        iouT = self.iouTensor(truB,preB)           # iouT (7*7,2)
        iouT = np.argmax(iouT, axis=1).astype(int) # (7*7)

        truB    = np.array([truB[i,j,:]  for i,j in enumerate(iouT)])
        preB    = np.array([preB[i,j,:]  for i,j in enumerate(iouT)])
        truConf = np.array([truConf[i,j] for i,j in enumerate(iouT)])
        preConf = np.array([preConf[i,j] for i,j in enumerate(iouT)])

        # Obj or noobj is actually only depend on truth
        objMask  = np.array([ max(i) for i in truCP])
        nobjMask = 1 - np.array([ max(i) for i in truCP])

        loss_ += sum(pow( (truB-preB), 2).sum(axis=1)    * objMask  ) * COORD 
        loss_ += sum(pow( (truConf- preConf) , 2)        * objMask  )
        loss_ += sum(pow( (truConf- preConf), 2)         * nobjMask ) * NOOBJ
        loss_ += sum(pow( (truCP- preCP), 2).sum(axis=1) * objMask  )
        return loss_

    def boxArea(self, box):
        return box[:,:,2]*box[:,:,3]

    def iouTensor(self, box1, box2):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        assert box1.shape == box2.shape == (S*S,B,4)
        
        fmin = np.vectorize(min)
        fmax = np.vectorize(max)

        minTop = fmin(box1[:,:,0]+0.5*box1[:,:,2],
                      box2[:,:,0]+0.5*box2[:,:,2])
        maxBot = fmax(box1[:,:,0]-0.5*box1[:,:,2],
                      box2[:,:,0]-0.5*box2[:,:,2])
        minR   = fmin(box1[:,:,1]+0.5*box2[:,:,3],
                      box1[:,:,1]+0.5*box2[:,:,3])
        maxL   = fmax(box1[:,:,1]-0.5*box2[:,:,3],
                      box1[:,:,1]-0.5*box2[:,:,3])
        # intersection
        inters = (minTop-maxBot).clip(min=0)* (minR-maxL).clip(min=0) 
        noZero = 0.000000001 # Return IOU and avoid devide zero
        return inters/ (self.boxArea(box1)+ self.boxArea(box2)- inters+ noZero)

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

    def train(self, ):
        model = self.build()
        loss = self.loss
        model.compile(optimizer=RMSprop(lr=0.001), loss=loss, metrics=['accuracy'])


if __name__ =='__main__':
    detector = YoloDetector()
    a = detector.encode(3, 22, 12, 123, 123)
    b = detector.encode(4, 22, 12, 123, 123)
    c = detector.encode(3, 22, 12, 123, 123)
    d = detector.encode(4, 23, 15, 111, 121)
    print detector.decode(a)
    print detector.decode(b)
    print detector.decode(c)
    print detector.loss(a, c)
    print detector.loss(a, b)
    print detector.loss(d, b)





