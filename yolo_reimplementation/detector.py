
from __future__ import division
import numpy as np

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
        self.classMap  =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


    def read_from(self):
        pass

    def encode(self, classid, cX, cY, boxW, boxH):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        assert int(classid) <= int(C-1)

        class_prob = np.zeros([S, S, C   ])
        confidence = np.zeros([S, S, B   ])
        boxes      = np.zeros([S, S, B, 4])

        # target the center grid
        gridX, gridY = W/S, H/S
        tarIdX, tarIdY = int(cX/gridX) , int(cY/gridY)

        # assign the true value
        class_prob[tarIdX, tarIdY, classid] = 1.0
        confidence[tarIdX, tarIdY, :      ] = 1.0

        # y,x,w,h , this order is really odor and killed me one day...
        boxes[tarIdX, tarIdY, :, 1] = (cX/gridX) - int(cX/gridX)
        boxes[tarIdX, tarIdY, :, 0] = (cY/gridY) - int(cY/gridY)
        boxes[tarIdX, tarIdY, :, 2] = np.sqrt(boxW/W)
        boxes[tarIdX, tarIdY, :, 3] = np.sqrt(boxH/H)

        output = np.concatenate([class_prob.flatten(), confidence.flatten(), boxes.flatten() ])

        return output

    def decode(self, predictions,threshold=0.2 ,only_objectness=0):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        probs = np.zeros((S,S,B,C))

        class_probs = np.reshape(output[0:S*S*C], (S,S,C))
        scales = np.reshape(output[S*S*C: S*S*(C+B)], (S,S,B)) # confidence
        boxes = np.reshape(output[S*S*(C+B):],(S,S,B,4))
        pass


    def loss(truY, preY):
        coord_1()+coord_2()+()+ noobj_()
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

        boxes[:,:,:,0] *= self.W
        boxes[:,:,:,1] *= self.H
        boxes[:,:,:,2] *= self.W
        boxes[:,:,:,3] *= self.H

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')

        filter_mat_boxes = np.nonzero(filter_mat_probs)
        print filter_mat_boxes
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
            result.append([self.classMap[classes_num_filtered[i]],boxes_filtered[i][1],boxes_filtered[i][0],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)


if __name__ =='__main__':
    detector = YoloDetect()
    for cX,cY in [(50,50),(50,150),(10,245),(440,440)]:
        print ('=======')
        print (cX,cY)

        a = detector.encode(3, cX, cY, 112, 12)

        print detector.interpret_output(a)

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


