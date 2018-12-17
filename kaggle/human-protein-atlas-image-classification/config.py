class DefaultConfigs(object):
    train_data = "C:/Users/kent/.kaggle/competitions/human-protein-atlas-image-classification/train/" # where is your train data
    test_data = "C:/Users/kent/.kaggle/competitions/human-protein-atlas-image-classification/test/"   # your test data
    weights = "hub_ckpts/"
    best_models = "hub_ckpts/best_models/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 50

config = DefaultConfigs()
