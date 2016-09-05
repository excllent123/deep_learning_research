### Basic Information
- Authors : Kent Chiu (Jin-Chun)
- Kickoff : 2016-09-05
- This Project is mainly focus on testing the deep learning algorithms on Image, Vedio and text-data.

### Dependence
- Anaconda - ENV.
- Theano/Tensorflow
- Keras
- Django
- Opencv

### Project Structure
- hub : the place to store the model, conf, log.
- doc : the place to store some documents
- dnn : the place to store the DNN structure. single dnn could have multiple conf files. however, dnn structures would constrain your conf parameters.
- sct : the place to store testing scripts.

### Usage Model (depend on argparse *args)
- single conf-files trainng test
- multi conf-files training test with same data (GridSearch Like)
- multi conf-files training test with different data (multi-process)

### Usage Flow
##### 1. New Structure Testing
- Define your model file in the dnn in the same manner
- Run *genConf.py* to generate new stucture of conf-folder

##### 2. Hyperparameter tuning
- Define your conf-files in the hub/conf/someFolder in the same manner with sample.json
- Run *trainDNN.py* -conf file1 file2 file3
- the modle would be save in hub/model
- the logging would be save in hub/log

##### 3. Detection or Parsers
- Run *testDNN.py* -conf file1 (single file)


### Future Work
- Docker integration
- WatchDog integration
- ...etc

### Lisence
- All rights reserved.
