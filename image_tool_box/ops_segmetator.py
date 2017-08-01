

import numpy as np


class ImgSegmentator:
	pass

def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else: 
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i-position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif(height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()    
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area =  maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea


def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            print (addMat[r])
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r-1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (int(maxArea[3]+1-maxArea[0]/abs(maxArea[1]-maxArea[2])),
            maxArea[2], maxArea[3], maxArea[1], maxArea[0])


def cropCircle(img):
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    print (tile_size)

    img = cv2.resize(img, dsize=tile_size)
            
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
    ff = np.zeros((gray.shape[0],gray.shape[1]), 'uint8') 
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0]+2,gray.shape[1]+2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 1)
    
    rect = maxRect(ff)

    print ('image before maxRect')
    imshow(ff)

    rectangle = [min(rect[0],rect[2]), max(rect[0],rect[2]), min(rect[1],rect[3]), max(rect[1],rect[3])]
    img_crop = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
    cv2.rectangle(ff,
        (min(rect[1],rect[3]),min(rect[0],rect[2])),
        (max(rect[1],rect[3]),max(rect[0],rect[2])),3,2)
    
    return [img_crop, rectangle, tile_size]

def Ra_space(img, Ra_ratio, a_threshold):
    '''
    '''
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            Ra[i*h+j, 1] = min(imgLab[i][j][1], a_threshold)
            
    Ra[:,0] /= max(Ra[:,0])
    Ra[:,0] *= Ra_ratio
    Ra[:,1] /= max(Ra[:,1])

    return Ra


def red_color_segmentation(img, Ra_ratio=0.95, a_threshold=155):
    #img = cv2.resize(img, (250,250))
    
    [img, rectangle_cropCircle, tile_size] = cropCircle(img)
    img = cv2.resize(img, dsize=tile_size)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    w = img.shape[0]
    h = img.shape[1]
    
    Ra = np.zeros((w * h, 2))
                   
    
    for i in range(w):
        for j in range(h):
            # R is a gradually-changing from center 
            R = math.sqrt((w / 2 - i) * (w / 2 - i) + (h / 2 - j) * (h / 2 - j))
            #print (R)
            # construct a (w,h,0) plane with centrol-low and surrounding high 
            Ra[i * h + j, 0] = R # imshow(Ra.reshape(w,h,2)[:,:,0])
            
            # construct a (w,h,1) plane with a_threshold as minimu + with origin-img
            Ra[i * h + j, 1] = min(img[i][j][1], a_threshold) # imshow(Ra.reshape(w,h,2)[:,:,1])
            
    Ra[:,0] /= max(Ra[:,0]) # normalization-by-sample 
    # imshow(Ra.reshape(w,h,2)[:,:,0])
    Ra[:,0] *= Ra_ratio
    #imshow(Ra.reshape(w,h,2)[:,:,0])
    Ra[:,1] /= max(Ra[:,1]) # normalization-by-sample
    #imshow(Ra.reshape(w,h,2)[:,:,1])
    #image_array_sample = shuffle(Ra, random_state=0)[:1000]
    #print (image_array_sample)
    # Use K-means to have 2 region 
    # 1-is dark 
    # the other is light 
    # one-image breaking-down into multiple rows of sample, with 2 extracting-feature-numbers
    # or to say, one-pixal are with 2-feature, one is a-channel numbers and one is a spatisal-feature
    # we use this 2 number to predict weather this is the region we want to segment 
    a_channel = np.reshape(Ra[:,1], (w,h))
                   
    g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag',
                    random_state = 0, init_params = 'kmeans')
    image_array_sample = shuffle(Ra, random_state=0)[:1000]
    g.fit(image_array_sample)
    
    # label is a one-dim tensor for better operations
    # labels already contain the information for region-propose
    labels = g.predict(Ra) # print (labels.shape) ==> w*h

    labels_2D = np.reshape(labels, (w,h))
    
    # get regionprops object 
    gg_labels_regions = measure.regionprops(labels_2D, intensity_image = a_channel)
    
    # get the intensity
    gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
    
    #print (gg_intensity)
    cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1
    #print (cervix_cluster)
    

    mask = np.zeros((w * h,1),'uint8')
    mask[labels==cervix_cluster] = 255
    mask_2D = np.reshape(mask, (w,h))
    # imshow(mask_2D)


    cc_labels = measure.label(mask_2D, background=0)
    #imshow(cc_labels)
    regions = measure.regionprops(cc_labels)
    areas = [prop.area for prop in regions]

    regions_label = [prop.label for prop in regions]
    largestCC_label = regions_label[areas.index(max(areas))]
    mask_largestCC = np.zeros((w,h),'uint8')
    mask_largestCC[cc_labels==largestCC_label] = 255

    img_masked = img.copy()
    img_masked[mask_largestCC==0] = (0,0,0)
    img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);
            
    _,thresh_mask = cv2.threshold(img_masked_gray,0,255,0)
    
    # dilate and erod for smooth        
    kernel = np.ones((9,9), np.uint8)
    thresh_mask = cv2.dilate(thresh_mask, kernel, iterations = 1)
    thresh_mask = cv2.erode(thresh_mask, kernel, iterations = 1)
    _, contours_mask, _ = cv2.findContours(thresh_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours_mask, key = cv2.contourArea, reverse = True)[0]
    #cv2.drawContours(img, main_contour, -1, 255, 3)

    x,y,w,h = cv2.boundingRect(main_contour)
    # imshow(img[y:y+h,x:x+w,:])
    
    # back-to-LAB-color-space
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img[y:y+h,x:x+w,:]