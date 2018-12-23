from hub_model.bsl_model import get_net, get_inceptionv3

class DefaultConfigs(object):
    train_data = "C:/Users/kent/.kaggle/competitions/human-protein-atlas-image-classification/train/" # where is your train data
    test_data = "C:/Users/kent/.kaggle/competitions/human-protein-atlas-image-classification/test/"   # your test data
    weights = "hub_ckpts/"
    best_models = "hub_ckpts/best_models/"
    submit = "hub_submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 50
    model = get_net(channels, num_classes)

class Incept3(DefaultConfigs):
    num_classes = 28
    img_weight = 299
    img_height = 299
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 50
    model_name = "inception_v3"
    model = get_inceptionv3(channels, num_classes)

class Incept3_02(DefaultConfigs):
    num_classes = 28
    img_weight = 299
    img_height = 299
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 100
    model_name = "inception_v3_02"
    model = get_inceptionv3(channels, num_classes)


config = Incept3_02()
