"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-01 22:42:30
 * @modify date 2022-02-26 23:57:42
 * @desc [description]
 """ 
 
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision.transforms.functional_tensor as F_t

import cv2
import json

import os

import pandas as pd

import random
from skimage.transform import rescale

import utils




class LSULungDataset(Dataset):

    def __init__(self, cfg, dataset_type, process_data = False):

        super(LSULungDataset, self).__init__()

        self.cfg = cfg

        self.datadir = os.path.join(cfg.DATA.ROOT_PATH, cfg.EXPERIMENT.DATASET)

        self.video_type = cfg.VIDEO_DATA.TYPE
        print(f"video_type = {self.video_type}")

        self.probe = cfg.DATA.PROBE
        print(f"probe = {self.probe}")

        self.numClasses = cfg.DATA.NUM_CLASS
        self.labels = np.arange(self.numClasses)
        self.label_names = cfg.DATA.LABEL_NAMES
        self.label_features = cfg.DATA.LABEL_FEATURES

        self.multi_task = cfg.MODEL.MULTI_TASK
        if self.multi_task:
            self.label2_names = cfg.DATA.LABEL2_NAMES
            self.label2_features = cfg.DATA.LABEL2_FEATURES

        label_json_path = os.path.join(self.datadir, self.cfg.DATA.LABEL_DICT)
        with open(label_json_path, 'r') as json_file:
            self.label_dict = json.load(json_file)

        self.video_names = np.sort(np.array(list(self.label_dict.keys())))


        self.excel_file_path = os.path.join(self.datadir, "LSU_Large_Dataset_videos (final).xlsx")
        self.patient_data_header, self.patient_data = self.getPatientData()

        #Select videos form excel patients
        self.patient_videos_to_include = [f"{p.split('-')[0]}/{p}.avi" for p in self.patient_data[:, self.patient_data_header.index("Video File Label (Pt # - file#)")]]


        self.processed_dataset_json_path = os.path.join(self.datadir, f"processed_dataset_files_{self.video_type}.json")

        if process_data:
            processed_dataset_dict = self._processData()
            processed_video_names = np.sort(np.array(list(processed_dataset_dict.keys())))
            
            labeled_video_names = ['.'.join(v.split('.')[:-1]) for v in self.video_names]

            if self.probe == "Linear":
                processed_video_names = [v.split('-')[0] for v in processed_video_names if '-C19-' not in v]
            elif self.probe == "C19-lung":
                processed_video_names = [v.replace('-C19-lung', '') for v in processed_video_names if '-C19-lung' in v]
            elif self.probe == "C19-cardiac":
                processed_video_names = [v.replace('-C19-cardiac', '')  for v in processed_video_names if '-C19-cardiac' in v]
            
            labeled_video_names_short = [v.split('-')[0] for v in labeled_video_names]
       
            #Check processed videos are all present in label_dict
            # assert np.array_equal(labeled_video_names, processed_video_names), "Missing video labels"
            assert np.array_equal(labeled_video_names_short, processed_video_names), "Missing video labels"
            
            return

        with open(self.processed_dataset_json_path, 'r') as json_file:
            self.processed_dataset_dict = json.load(json_file)


        self.image_size = cfg.DATA.IMAGE_RESIZE
        
        #video configs
        self.dataset_type = dataset_type
        self.clip_width = cfg.VIDEO_DATA.CLIP_WIDTH
        self.temp_sampling_method = cfg.VIDEO_DATA.TEMP_SAMPLING_METHOD

        self.shouldAugmentClip = cfg.DATA.AUGMENT and self.dataset_type == "train" #Augment only for Train dataset


        # self.test_split = cfg.DATA.TEST_SPLIT

        self.random_split_trial = cfg.DATA.RANDOM_SPLIT_TRIAL
        print(f'Using random generated split {self.random_split_trial}')


        # self.test_per = cfg.DATA.TEST_SPLIT


        self.foldsToIncludeList = []
        if self.dataset_type == "train":
            self.foldsToIncludeList = cfg.DATA.TRAIN_FOLDS
        elif self.dataset_type == "val":
            self.foldsToIncludeList = cfg.DATA.VAL_FOLDS
        elif self.dataset_type == "test":
            self.foldsToIncludeList = cfg.DATA.TEST_FOLDS
        else:
            raise ValueError(f"Unsupported cfg.EXPERIMENT.DATASET Fold = {cfg.EXPERIMENT.DATASET}!")
        
        print(f"foldsToIncludeList : {self.foldsToIncludeList}")

        self.videosToIncludeList = self.getVideosFromSplit()
        print(f'videosToIncludeList : {self.videosToIncludeList}')
        
        self.dataset_len = len(self.videosToIncludeList)

        self.SF_category_dict = utils.readJson(path = os.path.join(self.datadir, "SF_category_dict.json"))
        self.Disease_category_dict = utils.readJson(path = os.path.join(self.datadir, "Disease_category_dict.json"))

        self.upsampleData = self.cfg.DATA.UPSAMPLE and (self.dataset_type == "train" #UpSampleData only for Train dataset
                    or (self.cfg.DATA.UPSAMPLE_VAL and self.dataset_type == "val")) #UpSample Val if enabled
        if self.upsampleData:
            self.__upsampleData()

        self.multiClipVal = self.cfg.DATA.MULTI_CLIP_EVAL and self.dataset_type != "train" #MultiClipVal not to be done for Train dataset
        if self.multiClipVal:
            self.no_multi_clip = self.cfg.DATA.NO_MULTI_CLIP
            self.__multiClipsampleData()

    def getPatientData(self):

        patient_data = pd.read_excel(self.excel_file_path)
        header = list(patient_data.columns)
        patient_data = np.array(patient_data)
        
        #Get videos ids where SF ratio is empty
        non_sf_videos = np.argwhere(np.isnan(patient_data[:, header.index('S/F on date of exam')].astype(np.float32)))
        assert np.isnan(patient_data[non_sf_videos, header.index('S/F on date of exam')].astype(np.float32)).all(), "Error! Wrong indices picked."

        #Exluded non SF data videos
        patient_data = np.delete(patient_data, non_sf_videos, axis = 0)

        assert not np.isnan(patient_data[:, header.index('S/F on date of exam')].astype(np.float32)).any(), "Error! Wrong indices removed."

        return header, patient_data


    def getLabels(self):
        return self.labels

    def getLabelNames(self):
        return self.label_names

    def __upsampleData(self):
        #TODO-GRG: Re-randomly upsample at every training epoch - this should increase diversity
        
        # labels = np.array([np.argmax(self.label_dict[v][self.label_features]) for v in self.videosToIncludeList])
        if self.label_features[0] == "sf-ratio":
            labels = np.array([self.SF_category_dict[v] for v in self.videosToIncludeList])
        elif self.label_features[0] == "diagnosis":
            # labels = np.array([self.Disease_category_dict[v] for v in self.videosToIncludeList])
            
            # #Multi-label
            # """
            # #Note: As some videos are marked for multiple disease we are picking the first disease,
            # #  while considering upsampling. We can further optimize this
            # """
            # labels = np.array([self.Disease_category_dict[v] if isinstance(self.Disease_category_dict[v], int) else self.Disease_category_dict[v][0] for v in self.videosToIncludeList])

            #Multi-class (not multi-label)
            # labels = np.array([self.Disease_category_dict[v] for v in self.videosToIncludeList])

            #Map the 11 disease category to 7 categories
            disease_mapping_dict = {0: 0, 1: 1, 2: 5, 3: 4, 4: 5, 5: 5, 6: 2, 7: 2, 8: 3, 9: 6, 10: 5, 11: 5}
            labels = np.array([disease_mapping_dict[self.Disease_category_dict[v]] for v in self.videosToIncludeList])
        else:
            labels = np.array([np.argmax(self.label_dict[v]["lung-severity"]) for v in self.videosToIncludeList])

        assert len(labels) == len(self.videosToIncludeList), "Error! Label count mismatch."
        classes, count = np.unique(labels, return_counts = True)
        
        max_count = max(count)

        label_indices = []
        for c in classes:

            c_idx = np.where(labels == c)[0]
            assert np.unique(labels[c_idx]) == c, "Error! Wrong class index filtered."

            #Bug-GRG : Since we sample randomly some of the videos are never sampled/included. 
            # So, make sure to only sample additional required videos after including all videos at least once!
            #For the max count class, set replace to False as setting it True might exclude some samples from training
            # upsample_c_idx = np.random.choice(c_idx, size = max_count, replace = len(c_idx) < max_count)
            if len(c_idx) < max_count:
                # upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = len(c_idx) < max_count).tolist())
                upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = max_count > 2*len(c_idx)).tolist())
            else:
                upsample_c_idx = c_idx
            
            np.random.shuffle(upsample_c_idx)
            
            assert c_idx.shape == np.unique(upsample_c_idx).shape, "Error! Some videos where excluded on updampling."

            label_indices.extend(upsample_c_idx)

        assert len(label_indices) == max_count * len(classes)

        self.upsample_label_indices = label_indices

        print(f"Up-sampled classes to {max_count}; Dataset len increased to {len(self.upsample_label_indices)} from {self.dataset_len}")

        self.dataset_len = len(self.upsample_label_indices)

    def __multiClipsampleData(self):

        if self.upsampleData: #Apply multi-clip on top of upsampled data
            self.multiClip_sample_label_videos = np.array([[f"{self.videosToIncludeList[v]}-sf{f}" for f in range(self.no_multi_clip)] for v in self.upsample_label_indices]).flatten()
    
            assert len(self.multiClip_sample_label_videos) == len(self.upsample_label_indices)*self.no_multi_clip, "Error! Multi Clip Label count mismatch."
        else:
            self.multiClip_sample_label_videos = np.array([[f"{v}-sf{f}" for f in range(self.no_multi_clip)] for v in self.videosToIncludeList]).flatten()

            assert len(self.multiClip_sample_label_videos) == len(self.videosToIncludeList)*self.no_multi_clip, "Error! Multi Clip Label count mismatch."

        print(f"multiClip-sampled classes to {len(self.multiClip_sample_label_videos)} from {self.dataset_len}")

        self.dataset_len = len(self.multiClip_sample_label_videos)

    def __len__(self):
        return self.dataset_len

    def _temporalSampling(self, frames_list, sf_idx):
        
        if self.temp_sampling_method == "equi_temp_sampling":
            clip_stride = np.floor(len(frames_list)/self.clip_width)

            if self.multiClipVal:
                start_frame = np.arange(0, clip_stride, int(clip_stride/self.no_multi_clip))[sf_idx]
            else:
                start_frame = np.random.randint(clip_stride)
            
            clip_frame_ids = np.arange(start_frame, len(frames_list), clip_stride).astype(np.int)

            #Slice the clips
            clip_frame_ids = clip_frame_ids[:self.clip_width]

        elif self.temp_sampling_method == "rand_temp_sampling":

            if self.multiClipVal:
                clip_stride = 1
                start_frame = np.arange(0, len(frames_list), self.clip_width)[sf_idx]
            else:
                clip_stride = np.random.randint(1, 1+np.floor(len(frames_list)/self.clip_width))
                start_frame = np.random.randint(len(frames_list) - (self.clip_width*clip_stride)) if len(frames_list) > (self.clip_width*clip_stride) else 0
         
            clip_frame_ids = np.arange(start_frame, start_frame+(self.clip_width*clip_stride), clip_stride).astype(np.int)
        
        else:
            raise ValueError(f"Wrong cfg.DATA.TEMP_SAMPLING_METHOD = {self.temp_sampling_method}!")

        assert len(clip_frame_ids) == self.clip_width, "Clip width mismatch"

        return clip_frame_ids

    def __getitem__(self, index):
        
        sf_idx = -1
        if self.multiClipVal:
            video_name = self.multiClip_sample_label_videos[index]
            sf_idx = int(video_name.split("-")[-1].replace("sf", ""))
            video_name = "-".join(video_name.split("-")[:-1])
        elif self.upsampleData:
            up_index = self.upsample_label_indices[index]
            video_name = self.videosToIncludeList[up_index]
        else:
            video_name = self.videosToIncludeList[index]

        video_key = '.'.join(video_name.split('.')[:-1])
        
        if self.probe == "Linear":
            video_key = video_key
        elif self.probe == "C19-lung":
            video_key = video_key.split('-')[0]+'-C19-lung'
        elif self.probe == "C19-cardiac":
            video_key = video_key.split('-')[0]+'-C19-cardiac'
                
        frames_dict = self.processed_dataset_dict[video_key]
        frames_list = list(frames_dict.keys())

        clip_frame_ids = self._temporalSampling(frames_list, sf_idx)

        # clip = torch.Tensor(np.array([cv2.imread(frames_dict[frames_list[f]]) for f in clip_frame_ids])) #Dim - TxHxWxC
        
        if "flow" in self.cfg.VIDEO_DATA.TYPE:
            clip = self._loadFlow(clip_frame_ids, frames_dict, frames_list)    
        else:
            clip = self._loadRGB(clip_frame_ids, frames_dict, frames_list)    
    
        
        # # Resize the images
        # clip = F.interpolate(clip, size = self.image_size, mode = "bicubic", align_corners = True)
        # # clip = (clip - clip.min())/(clip.max() - clip.min())
        # clip = clip.clamp(min=0, max=255) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)


        if self.shouldAugmentClip:
            if "flow" in self.cfg.VIDEO_DATA.TYPE:
                clip = self.augmentFlowClip(clip)    
            else:
                clip = self.augmentRGBClip(clip)

        video_label = self.label_dict[video_name]

        label = []
        if self.label_features[0] == "sf-ratio":
            label = torch.tensor(self.SF_category_dict[video_name])
        elif self.label_features[0] == "diagnosis":
            # label = torch.tensor(self.Disease_category_dict[video_name])
            
            #Map the 11 disease category to 7 categories
            disease_mapping_dict = {0: 0, 1: 1, 2: 5, 3: 4, 4: 5, 5: 5, 6: 2, 7: 2, 8: 3, 9: 6, 10: 5, 11: 5}
            
            # #Multi-label 
            # label = torch.zeros(self.numClasses)
            # diseases = [self.Disease_category_dict[video_name]] if isinstance(self.Disease_category_dict[video_name], int) else self.Disease_category_dict[video_name]
            # for d in diseases:
            #     label[disease_mapping_dict[d]] = 1

            #Multi-class (not multi-label)
            label = torch.tensor(disease_mapping_dict[self.Disease_category_dict[video_name]])
        else:
            [label.extend(video_label[f]) for f in self.label_features]
            # if len(self.label_features) == 1 and self.label_features[0] == "lung-severity":
            if len(self.label_features) == 1:
                label = torch.tensor(np.argmax(label))
            else: #biomarkerLabels case
                label = torch.tensor(label).float()

        label_dict = {"lb_cls": label}

        if self.multi_task:
            label2 = []
            [label2.extend(video_label[f]) for f in self.label2_features]
            # if len(self.label_features) == 1 and self.label_features[0] == "lung-severity":
            if len(self.label2_features) == 1:
                label2 = torch.tensor(np.argmax(label2))
            else: #biomarkerLabels case
                label2 = torch.tensor(label2).float()
        
            label_dict["lb_2_cls"] = label2
            
        return video_key.split('/')[-1], clip, label_dict

    
    def _loadRGB(self, clip_frame_ids, frames_dict, frames_list):

        clip = torch.Tensor(np.array([cv2.resize(cv2.imread(frames_dict[frames_list[f]]), self.image_size) for f in clip_frame_ids])) #Dim - TxHxWxC
        clip = clip.permute(0,3,1,2) #NxHxWxC -> NxCxHxW
        clip = clip[:,0].unsqueeze(1) #Convert RGB to Grey as this dataset is Grey scale #TODO-GRG : Note this needs to be changed for other datasets

        if self.cfg.DATA.NUM_MASKS > 0:
            mask1 = cv2.resize(cv2.imread("/".join(frames_dict[frames_list[0]].split("/")[:-2] + [self.cfg.DATA.MASK1 + ".png"]))[:,:,0], 
                        self.image_size, interpolation=cv2.INTER_NEAREST)

            mask2 = cv2.resize(cv2.imread("/".join(frames_dict[frames_list[0]].split("/")[:-2] + [self.cfg.DATA.MASK2 + ".png"]))[:,:,0], 
                        self.image_size, interpolation=cv2.INTER_NEAREST)

            masks = np.vstack((mask1[np.newaxis,],mask2[np.newaxis,]))
            masks = torch.tensor(masks).repeat(clip.shape[0],1,1,1)
            clip = torch.column_stack((clip, masks))

        #Convert image from 0-255 to 0-1 range
        clip = clip/255.0

        return clip

    
    def _loadFlow(self, clip_frame_ids, frames_dict, frames_list):

        clip = torch.Tensor(np.array([cv2.resize(np.load(frames_dict[frames_list[f]]), self.image_size) for f in clip_frame_ids])) #Dim - TxHxWxC
        clip = clip.permute(0,3,1,2) #NxHxWxC -> NxCxHxW
        # clip = clip[:,0].unsqueeze(1) #Convert RGB to Grey as this dataset is Grey scale #TODO-GRG : Note this needs to be changed for other datasets

        # #Convert image from 0-255 to 0-1 range
        # clip = clip/255.0

        return clip

    
    def augmentRGBClip(self, clip):

        probHFlip = np.random.random()

        if probHFlip > 0.5:
            clip = torch.flip(clip, [-1])

        
        # probVFlip = np.random.random()

        # if probVFlip > 0.5:
        #     clip = torch.flip(clip, [-2])

        # probGB = np.random.random()

        # if probGB > 0.5:
        #     clip = self.gaussianBlur(clip)


        probST = np.random.random()

        if probST > 0.5:
            clip = self.spatialTransformAugClip(clip)
            
        
        # probGN = np.random.random()

        # if probGN > 0.5:
        #     clip = self.gaussianNoise(clip)

        probIntensity = np.random.random()

        if probIntensity > 0.5:
            # scale = np.random.choice([0.8, 0.9, 1.1], size = 1)[0]
            scale = np.random.uniform(low=0.8, high=1.2, size=(1,))[0]
            clip = clip * scale
            clip = clip.clamp(min=0.0, max=1.0)

        # probMls = np.random.random()

        # if probMls > 0.5:
        #     first_frame = clip[0,:,:,:]
        #     # first_frame = first_frame.cpu().detach().numpy()
        #     img, num_controls, scale = first_frame, 5, 0.05
        #     top, bottom = 0.05, 0.4
        #     _, height, width = img.shape
        #     coords_ht, coords_width = np.arange(top*height), height - np.arange(bottom*height)
        #     coords = np.concatenate((coords_ht, coords_width))
        #     random.shuffle(coords)
        #     contours_vec_p, contours_vec_q = [], []
        #     for i in range(num_controls):
        #         tmp = []
        #         tmp.append(coords[i])
        #         tmp.append(np.random.randint(0, width))
        #         contours_vec_p.append(tmp)
        #         f1, f2 = min(tmp[0] + scale*height, height), max(tmp[0] - scale*height,0) 
        #         n1 = int(random.uniform(f2, f1))
        #         f1, f2 = min(tmp[1] + scale*width, width), max(tmp[1] - scale*width,0) 
        #         n2 = int(random.uniform(f2, f1))
        #         contours_vec_q.append([n1, n2])

        #     p, q = np.array(contours_vec_p), np.array(contours_vec_q)
        #     # p, q = self.generate_pq(first_frame)
        #     clip = self.mls_rigid_deformation_inv(clip, p, q)

        clip = clip.clamp(min=0.0, max=1.0)

        return clip


    def augmentFlowClip(self, clip):

        # probHFlip = np.random.random()

        # if probHFlip > 0.5:
        #     clip = torch.flip(clip, [-1])

        
        # probVFlip = np.random.random()

        # if probVFlip > 0.5:
        #     clip = torch.flip(clip, [-2])

        # probGB = np.random.random()

        # if probGB > 0.5:
        #     clip = self.gaussianBlur(clip)


        # probST = np.random.random()

        # if probST > 0.5:
        #     clip = self.spatialTransformAugClip(clip)
            
        
        # probGN = np.random.random()

        # if probGN > 0.5:
        #     clip = self.gaussianNoise(clip)

        # probIntensity = np.random.random()

        # if probIntensity > 0.5:
        #     # scale = np.random.choice([0.8, 0.9, 1.1], size = 1)[0]
        #     scale = np.random.uniform(low=0.8, high=1.2, size=(1,))[0]
        #     clip = clip * scale
        #     clip = clip.clamp(min=0.0, max=1.0)

        # clip = clip.clamp(min=0.0, max=1.0)

        return clip

    def mls_rigid_deformation_inv(self, clip, p, q, alpha=1.0, density=1.0):
        ''' Rigid inverse deformation
        ### Params:
            * image - ndarray: original image
            * p - ndarray: an array with size [n, 2], original control points
            * q - ndarray: an array with size [n, 2], final control points
            * alpha - float: parameter used by weights
            * density - float: density of the grids
        ### Return:NN
            A deformed image.
        '''
        t, c, height, width = clip.shape

        q = q[:, [1, 0]]
        p = p[:, [1, 0]]

        # Make grids on the original image
        gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
        gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
        vy, vx = np.meshgrid(gridX, gridY)
        grow = vx.shape[0]  # grid rows
        gcol = vx.shape[1]  # grid cols
        ctrls = p.shape[0]  # control points

        # Compute
        reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
        reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
        reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

        w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
        w[w == np.inf] = 2**31 - 1
        pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
        phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
        qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
        qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
        reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
        reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
        reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
        reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

        mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) * 
                            reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                            reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
        reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
        neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
        neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
        reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
        mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
        mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
        Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2), 
                                mul_right.transpose(0, 3, 4, 1, 2)), 
                    axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
        Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
        Delta_verti[...,0,:] = -Delta_verti[...,0,:]
        B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
        try:
            inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]
            flag = False
        except np.linalg.linalg.LinAlgError:
            flag = True
            det = np.linalg.det(B)                                                          # [grow, gcol]
            det[det < 1e-8] = np.inf
            reshaped_det = det.reshape(grow, gcol, 1, 1)                                    # [grow, gcol, 1, 1]
            adjoint = B[:,:,[[1, 0], [1, 0]], [[1, 1], [0, 0]]]                             # [grow, gcol, 2, 2]
            adjoint[:,:,[0, 1], [1, 0]] = -adjoint[:,:,[0, 1], [1, 0]]                      # [grow, gcol, 2, 2]
            inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                          # [2, 2, grow, gcol]

        vqstar = reshaped_v - qstar                                                         # [2, grow, gcol]
        reshaped_vqstar = vqstar.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]

        # Get final image transfomer -- 3-D array
        temp = np.matmul(reshaped_vqstar.transpose(2, 3, 0, 1),
                        inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1)                   # [2, grow, gcol]
        norm_temp = np.linalg.norm(temp, axis=0, keepdims=True)                             # [1, grow, gcol]
        norm_vqstar = np.linalg.norm(vqstar, axis=0, keepdims=True)                         # [1, grow, gcol]
        transformers = temp / norm_temp * norm_vqstar + pstar                               # [2, grow, gcol]

        # Correct the points where pTwp is singular
        if flag:
            blidx = det == np.inf    # bool index
            transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
            transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

        # Removed the points outside the border
        transformers[transformers < 0] = 0
        transformers[0][transformers[0] > height - 1] = 0
        transformers[1][transformers[1] > width - 1] = 0

        for i in range(t):
            image = clip[i,:,:,:]
            image = torch.permute(image, (1, 2, 0))
            transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

            # Rescale image
            transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')
            transformed_image = torch.permute(torch.Tensor(transformed_image), (2, 0, 1))
            clip[i,:,:,:] = transformed_image
        return clip

    def gaussianNoise(self, clip):

        # sigma, = np.random.uniform(low=0.1, high=2, size=(1,)).tolist()
        # sigma, = np.random.uniform(low=0.001, high=0.005, size=(1,)).tolist()
        sigma, = np.random.uniform(low=0.0001, high=0.0005, size=(1,)).tolist()
        clip = clip + (sigma**0.5)*torch.randn(clip.shape)
        clip = clip.clamp(min=0.0, max=1.0)

        return clip

    def gaussianBlur(self, clip):

        # kernel_sz_x, kernel_sz_y = np.random.choice([3,5,7], 2).tolist()
        # sigma_x, sigma_y = np.random.uniform(low=0.1, high=2, size=(2,)).tolist()
        kernel_sz_x, kernel_sz_y = np.random.choice([3,5], 2).tolist()
        sigma_x, sigma_y = np.random.uniform(low=0.1, high=1, size=(2,)).tolist()
        clip = F_t.gaussian_blur(clip, [kernel_sz_x, kernel_sz_y], [sigma_x, sigma_y])

        return clip


    def spatialTransformAugClip(self, clip):
    
        t, c, h, w = clip.shape

        #Scale params
        sx, sy = np.random.uniform(low=0.8, high=1.3, size=(2,))
        # sx, sy = (1,1)

        #Scale params
        # tx, ty = np.random.uniform(low=-0.05, high=0.05, size=(2,))
        tx, = np.random.uniform(low=-0.05, high=0.05, size=(1,))
        ty, = np.random.uniform(low=-0.02, high=0.08, size=(1,))
        # tx, ty = (0,0) #(-0.2,0.2)

        #Rotation angle
        alpha = (np.pi/360.0)*np.random.uniform(low=-15, high=15, size=(1,))
        # alpha = (np.pi/360.0)*5

        R_scale = np.array([[sx,0],[0,sy]])
        R_rot = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]]).squeeze()
        T = np.array([tx,ty])

        theta = np.hstack((np.matmul(R_rot, R_scale),T[:,np.newaxis]))

        # Repeat the SAME transform tensor for all time-frames
        theta = torch.tensor(theta).float().unsqueeze(0).repeat(t,1,1)
        # # theta = theta.view(-1, 3, 4)
        # theta = theta.view(-1, 2, 3)

        # print(f"theta: {theta.detach().cpu().numpy()}")

        grid = F.affine_grid(theta, clip.size(), align_corners = True)
        aug_clip = F.grid_sample(clip, grid, align_corners = True)

        #Crop and resize
        cx, cy = (25, 25)
        aug_clip = aug_clip[:,:,cy:-cy,cx:-cx]

        # # Resize the images
        aug_clip = F.interpolate(aug_clip, size = (h,w), mode = "bicubic", align_corners = True)
        # clip = (clip - clip.min())/(clip.max() - clip.min())
        aug_clip = aug_clip.clamp(min=0.0, max=1.0) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)


        return aug_clip



    def getAnalysisSample(self):

        filename_array = []
        grey_array = []
        label_array = []
        
        for idx in range(self.__len__()):

            video_name, clip, label = self.__getitem__(idx)

            filename_array.append(video_name)
            grey_array.append(clip)
            label_array.append(label)
        
        return filename_array, grey_array, label_array



    def getVideosFromSplit(self):
        
        self.split_json_path = os.path.join(self.datadir, f'dataset_split_equi_class_{self.random_split_trial}.json')

        if not os.path.exists(self.split_json_path):
            self.generateDataSplitEquiClass(self.split_json_path)

        self.split_dict = utils.readJson(self.split_json_path)

        video_list = []
  
        [video_list.extend(self.split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in self.foldsToIncludeList]

        return video_list

    def generateDataSplitEquiClass(self, split_json_path):
        
        print(f"Generating data split : {split_json_path}")

        #Write split info to file
        infoFile = open(split_json_path.split(".")[0] + "_info.txt","w")
        infoFile.write(f"Generating data split : {split_json_path}\n\n")

        folds_list = self.cfg.DATA.TRAIN_FOLDS + self.cfg.DATA.VAL_FOLDS
        folds = len(folds_list)

        train_video_dict = {}
        test_video_dict = {}

        lung_severity = np.array([np.argmax(self.label_dict[l]["lung-severity"]) for l in self.video_names])

        print(f"Lung-severity Categories: \n {np.unique(lung_severity, return_counts = True)}")
        infoFile.write(f"\n\nLung-severity Categories: \n {np.unique(lung_severity, return_counts = True)}\n\n")


        pt_num = np.array([v.split('/')[0] for v in self.video_names])
    
        unique_pts, unique_pts_video_count = np.unique(pt_num, return_counts = True)
        # sort_idx = np.argsort(unique_pts_video_count)[::-1]
        # unique_pts = unique_pts[sort_idx]
        # unique_pts_video_count = unique_pts_video_count[sort_idx]
        # lung_severity = lung_severity[sort_idx]

        pts_details = []
        for idx, (pt, video_cnt) in enumerate(zip(unique_pts, unique_pts_video_count)):
            pt_idx = np.where(pt_num == pt)[0]
            pt_sev_list = lung_severity[pt_idx]
            # pt_mean_sev = np.ceil(pt_sev_list.mean() + pt_sev_list.std())
            pt_mean_sev = np.ceil(pt_sev_list.mean())
            
            sort_string = f"{video_cnt}_{pt_mean_sev}_{pt}"
            pts_details.append(sort_string)

        pts_details.sort(key = lambda x: (int(x.split("_")[0]), x.split("_")[1]))
        pts_details = pts_details[::-1]

        pts_details[0::4]

        foldA = pts_details[0::4]
        foldB = pts_details[1::4]
        foldC = pts_details[2::4]
        foldD = pts_details[3::4]


        

        self.patient_data_header 
        self.patient_data

        # self.excel_video_list = [f"{v[self.patient_data_header.index('patient_name')]}/{v[self.patient_data_header.index('Video File Label (Pt # - file#)')}"] for v in self.patient_data]
        self.excel_video_list = self.patient_data[:, self.patient_data_header.index('Video File Label (Pt # - file#)')]
        self.excel_video_list = [f"{p.split('-')[0]}/{p}.avi" for p in self.excel_video_list]

        SF_category = self.patient_data[:, self.patient_data_header.index('SF_Category')].astype(np.float32)

        self.SF_category_dict = {}
        for video, sf in zip(self.excel_video_list, SF_category):
            self.SF_category_dict[video] = int(sf)

        #Save SF_category_dict
        utils.writeJson(self.SF_category_dict, path = os.path.join(self.datadir, "SF_category_dict.json"))

        print(f"SF Categories: \n {np.unique(SF_category, return_counts = True)}")
        infoFile.write(f"\n\nSF Categories: \n {np.unique(SF_category, return_counts = True)}\n\n")

        self.Diseases_category = self.patient_data[:, self.patient_data_header.index('Diseases_Category')]
        # self.Diseases_category = [[int(i) for i in d.split(",")] for d in self.Diseases_category]
        all_diseases_category = []
        diseases_category = []
        for idx, d in enumerate(self.Diseases_category):
            tmp = []
            if isinstance(d, str):
                for i in d.split(','):
                    tmp.append(int(i))
                    all_diseases_category.append(int(i))
                diseases_category.append(tmp)
            else:
                diseases_category.append(d)
                all_diseases_category.append(d)

        print(f"Diseases Categories: \n {np.unique(all_diseases_category, return_counts = True)}")
        infoFile.write(f"\n\nDiseases Categories: \n {np.unique(all_diseases_category, return_counts = True)}\n\n")


        self.Disease_category_dict = {}
        for video, disease in zip(self.excel_video_list, diseases_category):
            self.Disease_category_dict[video] = disease

        #Save Disease_category_dict
        utils.writeJson(self.Disease_category_dict, path = os.path.join(self.datadir, "Disease_category_dict.json"))

        def getDetails(fold, video_names, infoFile, foldName = ""):
            
            pt_num = np.array([v.split('/')[0] for v in video_names])
    

            pt_names = []
            pt_videos = []

            for p_dt in fold:
                pt = p_dt.split('_')[-1]
                pt_names.append(pt)

                pt_idx = np.where(pt_num == pt)[0]

                pt_video = video_names[pt_idx]
                pt_videos.extend(pt_video)

            print_str = f"\n\nFold {foldName} Details:\n"
            print(print_str)
            infoFile.write(f"\n{print_str}\n")

            print_str = f"No of patients {len(pt_names)}"
            print(print_str)
            infoFile.write(f"\n{print_str}\n")
            print_str = f"No of videos {len(pt_videos)}"
            print(print_str)
            infoFile.write(f"\n{print_str}\n")

            pt_lung_severity = np.array([np.argmax(self.label_dict[l]["lung-severity"]) for l in pt_videos])
            print_str = f"Lung severity distribution:\n {np.unique(pt_lung_severity, return_counts = True)}"
            print(print_str)
            infoFile.write(f"\n{print_str}\n")


            pt_sf_category = np.array([self.SF_category_dict[v] for v in pt_videos])
            print_str = f"SF category distribution:\n {np.unique(pt_sf_category, return_counts = True)}"
            print(print_str)
            infoFile.write(f"\n{print_str}\n")

            pt_disease_category = np.array([self.Disease_category_dict[v] for v in pt_videos])
            
            pt_all_disease_category = []
            for d in pt_disease_category:
                if isinstance(d, int):
                    pt_all_disease_category.append(d)
                else:
                    pt_all_disease_category.extend(d)

            print_str = f"Disease category distribution:\n {np.unique(pt_all_disease_category, return_counts = True)}"
            print(print_str)
            infoFile.write(f"\n{print_str}\n")
        
            # return pt_names, pt_videos, pt_lung_severity, pt_sf_category, pt_disease_category

            fold_dict = {
                    f"{foldName}_pt_names": pt_names, 
                    f"{foldName}_pt_videos": pt_videos, 
                    f"{foldName}_pt_lung_severity": pt_lung_severity.tolist(), 
                    f"{foldName}_pt_sf_category": pt_sf_category.tolist(), 
                    f"{foldName}_pt_disease_category": pt_disease_category.tolist()
                }

            return fold_dict
            


        foldA_dict = getDetails(foldA, self.video_names, infoFile, foldName = "A")
        foldB_dict = getDetails(foldB, self.video_names, infoFile, foldName = "B")
        foldC_dict = getDetails(foldC, self.video_names, infoFile, foldName = "C")
        foldD_dict = getDetails(foldD, self.video_names, infoFile, foldName = "D")


        #Close split info file
        infoFile.close()

                
        assert len(np.unique(foldA_dict["A_pt_names"] + foldB_dict["B_pt_names"])) == len(foldA_dict["A_pt_names"]) + len(foldB_dict["B_pt_names"]), f"Error! patient overlap between fold A & B"
        assert len(np.unique(foldB_dict["B_pt_names"] + foldC_dict["C_pt_names"])) == len(foldB_dict["B_pt_names"]) + len(foldC_dict["C_pt_names"]), f"Error! patient overlap between fold B & C"
        assert len(np.unique(foldC_dict["C_pt_names"] + foldD_dict["D_pt_names"])) == len(foldC_dict["C_pt_names"]) + len(foldD_dict["D_pt_names"]), f"Error! patient overlap between fold C & D"
        assert len(np.unique(foldD_dict["D_pt_names"] + foldA_dict["A_pt_names"])) == len(foldD_dict["D_pt_names"]) + len(foldA_dict["A_pt_names"]), f"Error! patient overlap between fold D & A"

        split_dict = {
            "foldA": foldA_dict, 
            "foldB": foldB_dict, 
            "foldC": foldC_dict, 
            "foldD": foldD_dict, 
        }

        #Save splits
        utils.writeJson(split_dict, split_json_path)



    def _processData(self):
 
        video_exts = ["avi", "mp4", "gif", "mov", "mpeg"]

        if self.probe == "Linear":
            videos = np.hstack([[os.path.join(f,i) for i in os.listdir(os.path.join(self.datadir,f)) if i.split(".")[-1] not in video_exts and '-C19-' not in i and os.path.isdir(os.path.join(self.datadir,f,i))] for f in os.listdir(self.datadir) if os.path.isdir(os.path.join(self.datadir,f))])
        elif self.probe == "C19-lung":
            videos = np.hstack([[os.path.join(f,i) for i in os.listdir(os.path.join(self.datadir,f)) if i.split(".")[-1] not in video_exts and '-C19-lung' in i and os.path.isdir(os.path.join(self.datadir,f,i))] for f in os.listdir(self.datadir) if os.path.isdir(os.path.join(self.datadir,f))])
        elif self.probe == "C19-cardiac":
            videos = np.hstack([[os.path.join(f,i) for i in os.listdir(os.path.join(self.datadir,f)) if i.split(".")[-1] not in video_exts and '-C19-cardiac' in i and os.path.isdir(os.path.join(self.datadir,f,i))] for f in os.listdir(self.datadir) if os.path.isdir(os.path.join(self.datadir,f))])

        #sort videos
        videos.sort()

        print(f"No of videos = {len(videos)}")

        processed_dataset_dict = {}
        if os.path.exists(self.processed_dataset_json_path):
            
            with open(self.processed_dataset_json_path, 'r') as json_file:
                processed_dataset_dict = json.load(json_file)

        for video in videos:

            video_dir = os.path.join(self.datadir, video, self.video_type)

            frame_filenames = [f for f in os.listdir(video_dir) if '.png' in f or '.jpeg' in f or '.jpg' in f or '.JPG' in f]
            frame_filenames.sort()

            if video in processed_dataset_dict:

                processed_frames = list(processed_dataset_dict[video].keys())
                processed_frames.sort()

                if np.array_equal(frame_filenames, processed_frames):
                    continue
                else:
                    raise Exception(f"Frames mismatch for video = {video}!")

            else:
                print(f"Processing video : {video}")

                processed_frame_dict = {}
                for f in frame_filenames:
                    processed_frame_dict[f] = os.path.join(video_dir, f) 

                processed_dataset_dict[video] = processed_frame_dict



        #Save processed dataset dict
        processed_dataset_json = json.dumps(processed_dataset_dict, indent=4)
        f = open(self.processed_dataset_json_path, "w")
        f.write(processed_dataset_json)
        f.close()

        return processed_dataset_dict
