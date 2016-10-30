
from __future__ import division
import numpy as np

# =============================================================================
# encode : (Ground Truth Box | Image ) -> Ground Truth Y
# decode : Predict Tensor Y -> 
# the encode and decode are symmetric in logic, however since the prediction
# may contain multi-objects, the output object should be more complicated 
# =============================================================================

class Box(object):
    def __init__(self,numClass):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = 0
        self.probs = np.zeros((numClass,1))

class YoloDetect(object):
    def __init__(self, numClass=20, rawImgW=448, rawImgH=448, S=7, B=2):
        self.S = S
        self.B = B 
        self.C = numClass
        self.W = rawImgW
        self.H = rawImgH

    def read_from(self):
        pass

    def encode(self, classid, cX, cY, boxW, boxH):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        assert int(classid) <= int(C-1)

        # init skeleton
        confidence = np.zeros([S*S, B])
        norsqrtW  = np.zeros([S*S, B])
        norsqrtH  = np.zeros([S*S, B])
        offsetX    = np.zeros([S*S, B])
        offsetY    = np.zeros([S*S, B])
        class_prob = np.zeros([S*S, C])

        # Some para
        gridX, gridY = W/S, H/S
        tarGrid = int(cX/gridX)*S+int(cY/gridY) 

        # Asssign True Value
        confidence[tarGrid,:] = 1.0
        norsqrtW [tarGrid,:] = np.sqrt(boxW/W)
        norsqrtH [tarGrid,:] = np.sqrt(boxH/H)
        offsetX   [tarGrid,:] = (cX/gridX) - np.floor(cX/gridX)
        offsetY   [tarGrid,:] = (cY/gridY) - np.floor(cY/gridY)
        class_prob[tarGrid, classid] = 1.0
        
        # Flatten  
        class_prob = class_prob.flatten()
        confidence = confidence.transpose().flatten()
        tmp = np.dstack([offsetX,offsetY,norsqrtW,norsqrtH]).flatten()

        return np.concatenate([class_prob, confidence, tmp])


    def decode(self, predictions,threshold=0.2 ,only_objectness=0):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        boxes = []
        probs = np.zeros((S*S*B,C))
        for i in range(S*S):
            row = int(i / S)
            col = i % S
            for n in range(B):
                index = i*B+n
                p_index = S*S*C+i*B+n # i is the grid number [1, S*S]
                scale = predictions[p_index]
                box_index = S*S*(C+B) + (i*B+n)*4    

                # init box & recovery parameter
                new_box = Box(C)
                new_box.x = (predictions[box_index + 0] + col) / S * W
                new_box.y = (predictions[box_index + 1] + row) / S * H
                new_box.w = pow(predictions[box_index + 2], 2) * W
                new_box.h = pow(predictions[box_index + 3], 2) * H 

                # 
                for j in range(C):
                    class_index = i*C
                    prob = scale*predictions[class_index+j]
                    if(prob > threshold):
                        new_box.probs[j] = prob
                    else:
                        new_box.probs[j] = 0
                if(only_objectness):
                    new_box.probs[0] = scale
                boxes.append(new_box)
        return boxes



    def loss():
        pass


    def interpret_output(self,output):
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))# confidence
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
        
        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        
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
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result        


def convert_yolo_detections(predictions,C=20,B=2,S=7,threshold=0.2,only_objectness=0):
    boxes = []
    probs = np.zeros((S*S*B,C))
    for i in range(S*S):
        row = int(i / S)
        col = i % S
        for n in range(B):
            index = i*B+n
            p_index = S*S*C+i*B+n # i is the grid number [1, S*S]
            scale = predictions[p_index]
            box_index = S*S*(C+B) + (i*B+n)*4

            new_box = Box(C)
            new_box.x = (predictions[box_index + 0] + col) / S 
            new_box.y = (predictions[box_index + 1] + row) / S 
            new_box.h = pow(predictions[box_index + 2], 2) 
            new_box.w = pow(predictions[box_index + 3], 2) 

            for j in range(C):
                class_index = i*C
                prob = scale*predictions[class_index+j]
                if(prob > threshold):
                    new_box.probs[j] = prob
                else:
                    new_box.probs[j] = 0
            if(only_objectness):
                new_box.probs[0] = scale
            boxes.append(new_box)
    return boxes

if __name__ =='__main__':
    detector = YoloDetect()

    a = detector.encode(3, 22, 12, 421/448, 123/448)
    print a.shape
    print 7*7*30

    b = convert_yolo_detections(a)
    print b[0].x , b[0].y, b[0].w , b[0].h 
    b = detector.decode(a)
    print b[0].x , b[0].y, b[0].w , b[0].h 
    
    #assert b == [3,115,242, 100/448, 120/448]




# https://github.com/sunshineatnoon/Darknet.keras/blob/master/RunTinyYOLO.py
# 

def do_nms_sort(boxes,total,classes=20,thresh=0.5):
    for k in range(classes):
        for box in boxes:
            box.class_num = k
        sorted_boxes = sorted(boxes,cmp=prob_compare)
        for i in range(total):
            if(sorted_boxes[i].probs[k] == 0):
                continue
            boxa = sorted_boxes[i]
            for j in range(i+1,total):
                boxb = sorted_boxes[j]
                if(boxb.probs[k] != 0 and box_iou(boxa,boxb) > thresh):
                    boxb.probs[k] = 0
                    sorted_boxes[j] = boxb
    return sorted_boxes

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2;
    l2 = x2 - w2/2;
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2;
    r2 = x2 + w2/2;
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 or h < 0):
         return 0;
    area = w*h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

