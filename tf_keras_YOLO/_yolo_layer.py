# Author : Kent Chiu

from __future__ import division
import numpy as np
from keras.engine import Layer
import tensorflow as tf 

class YoloDetector(Layer):
    '''description: 
    this class inherit keras-layer and providse the connectivity of tensorflow 
    this class implement the customize yolo-loss & encode & decode from [paper]
    (https://pjreddie.com/media/files/papers/yolo.pdf) for multi-object recog.  
    '''
    def __init__(self, C=20, rImgW=448, rImgH=448, S=7, B=2, classMap=None):
        # C = number of class
        self.S = S
        self.B = B
        self.C = C
        self.W = rImgW
        self.H = rImgH
        self.iou_threshold=0.1
        if classMap:
            self.classMap = classMap
        else :
            self.classMap  = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train","tvmonitor"]

    def set_class_map(self, mappingList):
        assert type(mappingList)==list ; assert len(mappingList) == self.C
        self.classMap=mappingList

    def encode(self, annotations):
        ''' annotations : nested list contained
        '''
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H

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

        return np.concatenate([classProb.flatten(),confidence.flatten(),
                               boxes.flatten()])

    def traDim(self, pred, mode=3):
        ''' Dimension Transformation of Tensor '''
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H

        if mode == 3 :
            pred = np.array(pred)
            classProb  = np.reshape(pred[0:S*S*C]         , (S,S,C))
            confidence = np.reshape(pred[S*S*C: S*S*(C+B)], (S,S,B)) 
            boxes      = np.reshape(pred[S*S*(C+B):]      , (S,S,B,4))

        elif mode == 2 :
            classProb  = tf.reshape(pred[0:S*S*C]         , (S*S,C))
            confidence = tf.reshape(pred[S*S*C: S*S*(C+B)], (S*S,B)) 
            boxes      = tf.reshape(pred[S*S*(C+B):]      , (S*S,B,4))

        return classProb, confidence, boxes        

    def decode(self, prediction,threshold=8e-25 ,only_objectness=0):
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

    def loss(self, truY_, preY_, COORD=5. , NOOBJ=.5 , loss_=0 , batch_size=8):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H

        for batch in range(batch_size):
            truY = truY_[batch,:]
            preY = preY_[batch,:]

            truCP ,truConf, truB = self.traDim(truY, mode=2)
            preCP ,preConf, preB = self.traDim(preY, mode=2)    

            # Select for responsible box which with max IOU
            iouT = self.iouTensor(truB,preB)           # iouT (7*7,2)
            iouT = tf.argmax(iouT, dimension=1) # (7*7)    

            # tf.cast(x, dtype, name=None)
            def slec_Box(raw , iouT):
                for i in range(S*S):
                    j = iouT[i]    

                    # flatten input 2D 
                    raw = tf.reshape(raw,[-1])     

                    # cast the idx to the right tyle
                    idx = tf.cast(tf.constant([0,1,2,3]), tf.int64)
                    idx_flattened = idx + (i*B*4+j)              
                    yield tf.gather(raw, idx_flattened)    

            def slec_conf(raw , iouT):
                for i in range(S*S):
                    j = iouT[i]    

                    # flatten input 2D 
                    raw = tf.reshape(raw,[-1])     

                    # cast the idx to the right tyle
                    idx = tf.cast(tf.constant([0]), tf.int64)
                    idx_flattened = idx + (i*B+j)              
                    yield tf.gather(raw, idx_flattened)
            
            truB    = tf.pack ([ a for a in slec_Box (truB    , iouT)] )
            preB    = tf.pack ([ a for a in slec_Box (preB    , iouT)] )
            truConf = tf.pack ([ a for a in slec_conf(truConf , iouT)] )
            preConf = tf.pack ([ a for a in slec_conf(preConf , iouT)] )    

            # Obj or noobj is actually only depend on truth
            # truCP = (S*S,C)
            def max_tf(raw):
                for i in range(S*S):
                    tmp = raw[i,:]
                    tmp = tf.reduce_max(tmp)
                    yield tmp    

            objMask  = tf.pack([ a for a in max_tf(truCP) ])
            nobjMask = 1 - objMask    

            loss_ += tf.reduce_sum(tf.reduce_sum(tf.pow(truB-preB, 2), 1)   * objMask  ) * COORD 
            loss_ += tf.reduce_sum(tf.pow(truConf- preConf, 2)              * objMask  )
            loss_ += tf.reduce_sum(tf.pow(truConf- preConf, 2)              * nobjMask ) * NOOBJ
            loss_ += tf.reduce_sum(tf.reduce_sum(tf.pow(truCP- preCP, 2), 1)* objMask  )
        return loss_ / batch_size

    def boxArea(self, box):
        return box[:,:,2]*box[:,:,3]

    def iouTensor(self, box1, box2):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        assert box1.get_shape() == box2.get_shape() == (S*S,B,4)
        
        minTop = tf.minimum(box1[:,:,0]+0.5*box1[:,:,2],
                      box2[:,:,0]+0.5*box2[:,:,2])
        maxBot = tf.maximum(box1[:,:,0]-0.5*box1[:,:,2],
                      box2[:,:,0]-0.5*box2[:,:,2])
        minR   = tf.minimum(box1[:,:,1]+0.5*box2[:,:,3],
                      box1[:,:,1]+0.5*box2[:,:,3])
        maxL   = tf.maximum(box1[:,:,1]-0.5*box2[:,:,3],
                      box1[:,:,1]-0.5*box2[:,:,3])
        # intersection

        #tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)

        inters = tf.clip_by_value( minTop-maxBot, clip_value_min=0, clip_value_max=999)* \
                 tf.clip_by_value( minR-maxL    , clip_value_min=0, clip_value_max=999)
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





