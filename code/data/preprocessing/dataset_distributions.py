
import numpy as np

import json

import os

import pandas as pd


import sys
sys.path.append("./code")
import utils


def getPatientData(excel_file_path):

    patient_data = pd.read_excel(excel_file_path)
    header = list(patient_data.columns)
    patient_data = np.array(patient_data)
    
    #Get videos ids where SF ratio is empty
    non_sf_videos = np.argwhere(np.isnan(patient_data[:, header.index('S/F on date of exam')].astype(np.float32)))
    assert np.isnan(patient_data[non_sf_videos, header.index('S/F on date of exam')].astype(np.float32)).all(), "Error! Wrong indices picked."

    #Exluded non SF data videos
    patient_data = np.delete(patient_data, non_sf_videos, axis = 0)

    assert not np.isnan(patient_data[:, header.index('S/F on date of exam')].astype(np.float32)).any(), "Error! Wrong indices removed."

    return header, patient_data

def getVideosFromSplit(datadir, foldsToIncludeList = ["A", "B", "C", "D"], random_split_trial = "R1"):
        
        split_json_path = os.path.join(datadir, f'dataset_split_equi_class_{random_split_trial}.json')

        split_dict = utils.readJson(split_json_path)

        video_list = []
  
        [video_list.extend(split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in foldsToIncludeList]

        return video_list

lung_severity_labels = [ 'score-0', 'score-1', 'score-2', 'score-3', ]
sf_raio_labels = [ '> 430', '275-430', '180-275', '<180', ]
disease_labels =  [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others', ]
  

if __name__ == "__main__":

    datadir = "/data1/datasets/LSU-Large-Dataset"

    video_list = getVideosFromSplit(datadir)
    assert len(video_list) == 718, "Error! No of videos mismatch."

    excel_file_path = os.path.join(datadir, "LSU_Large_Dataset_videos (final).xlsx")

    patient_data_header, patient_data = getPatientData(excel_file_path)

    excel_video_list = patient_data[:, patient_data_header.index('Video File Label (Pt # - file#)')]
    excel_video_list = [f"{p.split('-')[0]}/{p}.avi" for p in excel_video_list]

    lung_severity_dict = utils.readJson(path = os.path.join(datadir, "user_label_ggare_2.json"))

    video_lung_severity_category = [np.argmax(lung_severity_dict[v]["lung-severity"]) for v in video_list]
    assert len(video_lung_severity_category) == 718, "Error! No of videos mismatch."
    assert np.array_equal(np.unique(video_lung_severity_category), np.arange(len(lung_severity_labels))), "Error! No of class mismatch."
    


    # SF_category = patient_data[:, patient_data_header.index('SF_Category')].astype(np.float32)

    SF_category_dict = utils.readJson(path = os.path.join(datadir, "SF_category_dict.json"))

    # print(f"SF Categories: \n {np.unique(SF_category, return_counts = True)}")

    video_sf_category = [SF_category_dict[v] for v in video_list]
    assert len(video_sf_category) == 718, "Error! No of videos mismatch."
    assert np.array_equal(np.unique(video_sf_category), np.arange(len(sf_raio_labels))), "Error! No of class mismatch."
    


    # Diseases_category = patient_data[:, patient_data_header.index('Diseases_Category')]

    Disease_category_dict = utils.readJson(path = os.path.join(datadir, "Disease_category_dict.json"))

    # print(f"Diseases Categories: \n {np.unique(all_diseases_category, return_counts = True)}")

    disease_mapping_dict = {0: 0, 1: 1, 2: 5, 3: 4, 4: 5, 5: 5, 6: 2, 7: 2, 8: 3, 9: 6, 10: 5, 11: 5}
            
    #Multi-class (not multi-label)
    video_diseases_category = [disease_mapping_dict[Disease_category_dict[v]] for v in video_list]
    assert len(video_diseases_category) == 718, "Error! No of videos mismatch."
    assert np.array_equal(np.unique(video_diseases_category), np.arange(len(disease_labels))), "Error! No of class mismatch."

    vidoes_per_patient = [v.split('/')[0] for v in video_list]
    assert len(vidoes_per_patient) == 718, "Error! No of videos mismatch."
   
    patient_names, per_patient_video_count = np.unique(vidoes_per_patient, return_counts = True)
    per_patient_video_count.sort()
    per_patient_video_count = np.array(per_patient_video_count)[::-1]
    
    print(per_patient_video_count)

    print(np.unique(video_lung_severity_category, return_counts = True))
    print(np.unique(video_sf_category, return_counts = True))
    print(np.unique(video_diseases_category, return_counts = True))

    print("Finished!")
