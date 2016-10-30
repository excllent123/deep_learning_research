
from __future__ import division
import numpy as np

# =============================================================================
# Backward : (Ground Truth Box | Image ) -> Ground Truth Y
# =============================================================================


class Backward(object):
    def __init__(self, numClass=20, rawImgW=448, rawImgH=448, S=7, B=2):
        self.S = S
        self.B = B 
        self.C = numClass
        self.W = rawImgW
        self.H = rawImgH

    def trans_format(self, ):
        pass

    def encode(self, classid, cX, cY, norBox_W, norBox_H):
        '''
        norBox_W, norBox_H : bbox w/h noremalized by rawImg w/h
        where cX,cY is center of the bonding box        
        -------------------------------------------------------------
        Input : annotation information and bounding box size
        Output : tensor (S, S, (B*5+C)) 
        '''

        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        assert int(classid) <= int(C-1)

        # init the output tensor container eight np  or tf style
        confidence = np.zeros((S*S, B))
        box_sqrtW  = np.zeros((S*S, B))
        box_sqrtH  = np.zeros((S*S, B))
        offsetX    = np.zeros((S*S, B))
        offsetY    = np.zeros((S*S, B))
        class_prob = np.zeros((S*S, C))
        # since B = 2, if we want to have same length, 
        # we could devide the tensor to 11 tensors (2*5+C)

        # feed value in target-grid that contains the center of bbox

        gridX, gridY = W/S, H/S
        # The centerX/gridX is the scale the x unit from 1 to gridX
        # The 2D(S*S) array is serielized to 1D

        tarGrid = int(cX/gridX)*S+int(cY/gridY) 

        # Both normalize and offset is to encode tensor in [0,1]
        confidence[tarGrid,:] = 1.0
        box_sqrtW [tarGrid,:] = np.sqrt(norBox_W)
        box_sqrtH [tarGrid,:] = np.sqrt(norBox_H)
        offsetX   [tarGrid,:] = (cX/gridX) - np.floor(cX/gridX)
        offsetY   [tarGrid,:] = (cY/gridY) - np.floor(cY/gridY)
        class_prob[tarGrid, classid] = 1.0
        
        return np.concatenate([])
        






