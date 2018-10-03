import numpy as np
import pandas as pd  
import pydicom
from tqdm import tqdm

def get_metadata_per_patient(file_pth, attribute):
    '''
    Given a patient ID, return useful meta-data from the corresponding dicom image header.
    Return: 
    attribute value
    '''
    # get dicom image
    dcmdata =  pydicom.read_file(file_pth)
    # extract attribute values
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value


def get_hook_df():
    train_label = pd.read_csv('C:/Users/kent/.kaggle/competitions/rsna-pneumonia-detection-challenge/stage_1_train_labels.csv')
    trian_img_dir = 'C:/Users/kent/.kaggle/competitions/rsna-pneumonia-detection-challenge/stage_1_train_images/'    

    train_label['pth'] = train_label.patientId.apply(lambda x : trian_img_dir+x+'.dcm')     

    attributes = ['PatientSex', 'PatientAge', 'ViewPosition' ]
    for i in attributes:
        train_label[i]  = train_label['pth'].apply(lambda x: get_metadata_per_patient(x, i))

    train_label = train_label.fillna(0)
    return train_label


hood_df = get_hook_df()

res = []
for i in tqdm(hood_df['pth']):
	img = pydicom.dcmread(i).pixel_array
	img = img.astype("float32")
	img /=255.
	
	save_pth = i.replace('C:/Users/kent/.kaggle/competitions/rsna-pneumonia-detection-challenge/stage_1_train_images/', '')
	save_pth = 'train_img/' + save_pth.replace('.dcm',".np")
	np.save(save_pth, img )
	res.append(save_pth)
hood_df['pth'] = res
hood_df.to_feather('hook_df2.feather')