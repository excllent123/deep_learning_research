
Test Record.

### /sct_test/test_yolo_preprocess & YOLO/yolo_preprocess
(2016-11-10)
build the preprocesser for image batch generater
build the unit-test


### sct-run-xxx.py (2016-11-11)
found out that type-failure in loss function 
need to shift numpy-loss to tf-loss

### tf-yolo.py (2016-11-12)
implement the tf-loss
test-pass


### tfkeras.py (2016-11-13)
Could be work just the loss is not able to convergence.

### tf-keras-20161113.py (2016-11-13)
This slightly change the way for define a model.


### tf-keras-test (2016-11-14)
This is a test version of keras model for tf.sess
The result is confirmed that keras model did work at leat 
for a cetain arrangement like tf-keras-20161113.py

Here is a hint from [Here](https://groups.google.com/forum/#!searchin/darknet/converge%7Csort:relevance/darknet/hhFgA-cY9Ko/oyK--gRMBQAJ)

YOLO was pre-trained on imagenet which includes 1.2 million images and 1000 different classes.  Unless you think your data set is comparable to that magnitude of data, you should only be using your custom data set to fine-tune the pretrained YOLO weights.   

### find fuxx bug 
be aware of sess(init)

### Adding the tf-keras-20161114
modification of the yolo-preprocessor 
now need to add detector explicitly 

### test tensorflow saving mechnism => keep find no file 
????????????
UNSOVLE
????????????