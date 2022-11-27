"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-01 22:42:30
 * @modify date 2021-11-16 21:43:22
 * @desc [description]
 """ 
 
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import cv2
import json

import os

import utils




class VideoListDataset(Dataset):

    def __init__(self, cfg, dataset_type, label_dict):

        super(VideoListDataset, self).__init__()

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

        self.label_dict = label_dict

        self.video_names = np.sort(np.array(list(self.label_dict.keys())))

        self.videos = ['.'.join(v.split('.')[:-1]) for v in self.video_names]

        self.processed_dataset_dict = self._processData(self.videos)
        processed_video_names = np.sort(np.array(list(self.processed_dataset_dict.keys())))
        
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
        


        self.image_size = cfg.DATA.IMAGE_RESIZE
        
        #video configs
        self.dataset_type = dataset_type
        self.clip_width = cfg.VIDEO_DATA.CLIP_WIDTH

        self.shouldAugmentClip = cfg.DATA.AUGMENT and self.dataset_type == "train" #Augment only for Train dataset


        self.test_split = cfg.DATA.TEST_SPLIT

        self.random_split_trial = cfg.DATA.RANDOM_SPLIT_TRIAL
        print(f'Using random generated split {self.random_split_trial}')


        self.test_per = cfg.DATA.TEST_SPLIT



        self.videosToIncludeList = self.video_names
        print(f'videosToIncludeList : {self.videosToIncludeList}')
        
        self.dataset_len = len(self.videosToIncludeList)

        self.upsampleData = self.cfg.DATA.UPSAMPLE and self.dataset_type == "train" #UpSampleData only for Train dataset
        if self.upsampleData:
            self.__upsampleData()

        self.multiClipVal = self.cfg.DATA.MULTI_CLIP_EVAL and self.dataset_type != "train" #MultiClipVal not to be done for Train dataset
        if self.multiClipVal:
            self.no_multi_clip = self.cfg.DATA.NO_MULTI_CLIP
            self.__multiClipsampleData()

    def getLabels(self):
        return self.labels

    def getLabelNames(self):
        return self.label_names

    def __upsampleData(self):

        # labels = np.array([np.argmax(self.label_dict[v][self.label_features]) for v in self.videosToIncludeList])
        labels = np.array([np.argmax(self.label_dict[v][self.label_features[0]]) for v in self.videosToIncludeList])

        assert len(labels) == len(self.videosToIncludeList), "Error! Label count mismatch."
        classes, count = np.unique(labels, return_counts = True)
        
        max_count = max(count)

        label_indices = []
        for c in classes:

            c_idx = np.where(labels == c)[0]
            assert np.unique(labels[c_idx]) == c, "Error! Wrong class index filtered."

            #For the max count class, set replace to False as setting it True might exclude some samples from training
            upsample_c_idx = np.random.choice(c_idx, size = max_count, replace = len(c_idx) < max_count)
            
            label_indices.extend(upsample_c_idx)

        assert len(label_indices) == max_count * len(classes)

        self.upsample_label_indices = label_indices

        print(f"Up-sampled classes to {max_count}; Dataset len increased to {len(self.upsample_label_indices)} from {self.dataset_len}")

        self.dataset_len = len(self.upsample_label_indices)

    def __multiClipsampleData(self):

        self.multiClip_sample_label_videos = np.array([[f"{v}-sf{f}" for f in range(self.no_multi_clip)] for v in self.videosToIncludeList]).flatten()

        assert len(self.multiClip_sample_label_videos) == len(self.videosToIncludeList)*self.no_multi_clip, "Error! Multi Clip Label count mismatch."

        print(f"multiClip-sampled classes to {len(self.multiClip_sample_label_videos)} from {self.dataset_len}")

        self.dataset_len = len(self.multiClip_sample_label_videos)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):

        if self.upsampleData:
            up_index = self.upsample_label_indices[index]
            video_name = self.videosToIncludeList[up_index]
        elif self.multiClipVal:
            video_name = self.multiClip_sample_label_videos[index]
            sf_idx = int(video_name.split("-")[-1].replace("sf", ""))
            video_name = "-".join(video_name.split("-")[:-1])
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

        clip_stride = np.floor(len(frames_list)/self.clip_width)

        if self.multiClipVal:
            start_frame = np.arange(0, clip_stride, int(clip_stride/self.no_multi_clip))[sf_idx]
        else:
            start_frame = np.random.randint(clip_stride)
        
        clip_frame_ids = np.arange(start_frame, len(frames_list), clip_stride).astype(np.int)

        #Slice the clips
        clip_frame_ids = clip_frame_ids[:self.clip_width]

        assert len(clip_frame_ids) == self.clip_width, "Clip width mismatch"

        # clip = torch.Tensor([cv2.imread(frames_dict[frames_list[f]]) for f in clip_frame_ids]) #Dim - TxHxWxC
        clip = torch.Tensor([cv2.resize(cv2.imread(frames_dict[frames_list[f]]), self.image_size) for f in clip_frame_ids]) #Dim - TxHxWxC
        clip = clip.permute(0,3,1,2) #NxHxWxC -> NxCxHxW
        clip = clip[:,0].unsqueeze(1) #Convert RGB to Grey as this dataset is Grey scale #TODO-GRG : Note this needs to be changed for other datasets

        # # Resize the images
        # clip = F.interpolate(clip, size = self.image_size, mode = "bicubic", align_corners = True)
        # # clip = (clip - clip.min())/(clip.max() - clip.min())
        # clip = clip.clamp(min=0, max=255) #With bicuic interpolation mode, values can over shoot 255 (refer PyTorch Docs)

        #Convert image from 0-255 to 0-1 range
        clip = clip/255.0


        if self.shouldAugmentClip:
            clip = self.augmentClip(clip)

        video_label = self.label_dict[video_name]

        label = []
        [label.extend(video_label[f]) for f in self.label_features]
        label = torch.tensor(np.argmax(label))
            
        return video_key.split('/')[-1], clip, label


    def augmentClip(self, clip):

        probHFlip = np.random.random()

        if probHFlip > 0.5:
            clip = torch.flip(clip, [-1])

        
        # probVFlip = np.random.random()

        # if probVFlip > 0.5:
        #     clip = torch.flip(clip, [-2])


        probST = np.random.random()

        if probST > 0.5:
            clip = self.spatialTransformAugClip(clip)


        probIntensity = np.random.random()

        if probIntensity > 0.5:
            # scale = np.random.choice([0.8, 0.9, 1.1], size = 1)[0]
            scale = np.random.uniform(low=0.8, high=1.2, size=(1,))[0]
            clip = clip * scale
            clip = torch.clamp(clip, min=0, max=1)

        return clip



    def spatialTransformAugClip(self, clip):
    
        t, c, h, w = clip.shape

        #Scale params
        sx, sy = np.random.uniform(low=0.8, high=1.3, size=(2,))
        # sx, sy = (1,1)

        #Scale params
        tx, ty = np.random.uniform(low=-0.05, high=0.05, size=(2,))
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
        
        split_json_path = os.path.join(self.datadir, f'dataset_split_equi_class_{self.random_split_trial}.json')

        if not os.path.exists(split_json_path):
            self.generateDataSplitEquiClass(split_json_path)

        with open(split_json_path, 'r') as json_file:
            split_dict = json.load(json_file)

        video_list = []
        if self.dataset_type != 'test':
            video_fold_dict = split_dict['train']
            video_dict = []
            [[video_list.extend(v) for k,v in severity_videos.items() if k in self.foldsToIncludeList] for severity_videos in video_fold_dict.values()]

            assert len(video_list) == np.array([[len(v) for k,v in severity_videos.items() if k in self.foldsToIncludeList] for severity_videos in video_fold_dict.values()]).sum(), "Error! Videos selected from wroing split."
        else:
            video_dict = split_dict[self.dataset_type]

            [video_list.extend(s) for s in video_dict.values()]

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

        lung_severity = np.array([np.argmax(self.label_dict[l][self.label_features[0]]) for l in self.video_names])

        pt_num = np.array([v.split('(')[0] for v in self.video_names])
    
        unique_pts = np.unique(pt_num)
    
        pts_with_same_sev = []
        for pt in unique_pts:
            pt_idx = np.where(pt_num == pt)[0]
            pt_sev = lung_severity[pt_idx]
            if len(np.unique(pt_sev)) == 1:

                if len(pt_idx) > 2:
                    continue

                pts_with_same_sev.append(pt)


        #Process all severity images
        for severity in np.unique(lung_severity):
            video_idx = np.where(lung_severity == severity)

            video_list = self.video_names[video_idx]
            
            assert np.unique([np.argmax(self.label_dict[l][self.label_features[0]]) for l in video_list]) == severity

            np.random.shuffle(video_list)

            n = len(video_list)

            no_test_videos = max(int(np.round(n*self.test_per)), 1)

            #Make test videos count even
            no_test_videos += 1 if no_test_videos % 2 == 1 else 0
            
            no_train_videos = n - no_test_videos

            
            print(f"{severity} : Total no of Videos = {n}; [Train = {no_train_videos}; Test = {no_test_videos}]")
            infoFile.write(f"{severity} : Total no of Videos = {n}; [Train = {no_train_videos}; Test = {no_test_videos}]\n")

            #Select Test video's from unique severity pt list:
            unique_sev_video_list = [v for v in video_list if v.split('(')[0] in pts_with_same_sev]
            assert len(unique_sev_video_list) >= no_test_videos, "Error! Videos for unique severity is less than needed for test set."
            
            unique_sev_pts_list = np.unique([v.split('(')[0] for v in unique_sev_video_list])

            sel_test_pts = np.random.choice(unique_sev_pts_list, size = int(no_test_videos/2), replace = False) #Note: For now, we have 2 videos per patients
            test_videos = [v for v in unique_sev_video_list if v.split('(')[0] in sel_test_pts]

            test_video_dict[str(severity)] = test_videos

            non_test_videos = [v for v in video_list if v not in test_video_dict[str(severity)]]
            np.random.shuffle(non_test_videos)

            assert len(non_test_videos) == no_train_videos, "Error! Train video count mismatch."
        
            fold_len = int(np.floor(no_train_videos/folds))

            train_folds_dict = {}
            for fold_id in range(folds):
                train_folds_dict[folds_list[fold_id]] = non_test_videos[fold_len*fold_id:fold_len*(fold_id+1)]
            
            #Append the remaining videos 
            random_fold_list = folds_list.copy()
            np.random.shuffle(random_fold_list) #Randomize the fold order - inorder to create equally weighted folds

            f_c = 0
            for video in non_test_videos[fold_len*(fold_id+1):]:
                train_folds_dict[random_fold_list[f_c]].append(video)
                f_c+=1
            
            train_video_dict[str(severity)] = train_folds_dict

            assert no_test_videos == len(test_video_dict[str(severity)])
            assert no_train_videos == sum([len(l) for l in train_video_dict[str(severity)].values()])

            print(f"{severity} : Train-folds - {[len(l) for l in train_video_dict[str(severity)].values()]}")
            infoFile.write(f"{severity} : Train-folds - {[len(l) for l in train_video_dict[str(severity)].values()]}\n\n")

        #Close split info file
        infoFile.close()

        test_pts = np.unique(np.hstack([[v.split('(')[0] for v in v_list] for v_list in test_video_dict.values()]))

        train_pts = []
        [[train_pts.extend([v.split('(')[0] for v in v_list]) for v_list in l.values()] for l in train_video_dict.values()]
        train_pts = np.unique(train_pts)

        for t_pt in test_pts:
            assert t_pt not in train_pts, f"Error! test patient {t_pt} is in train set"

        split_dict = {"train": train_video_dict, "test": test_video_dict}

        #Save splits
        split_json = json.dumps(split_dict, indent=4)
        f = open(split_json_path,"w")
        f.write(split_json)
        f.close()



    def _processData(self, videos):
 
        video_exts = ["avi", "mp4", "gif", "mov", "mpeg"]

        if self.probe == "Linear":
            videos = np.hstack([i for i in videos if i.split(".")[-1] not in video_exts and '-C19-' not in i and os.path.isdir(os.path.join(self.datadir,i))])
        elif self.probe == "C19-lung":
            videos = np.hstack([i for i in videos if i.split(".")[-1] not in video_exts and '-C19-lung' in i and os.path.isdir(os.path.join(self.datadir,i))])
        elif self.probe == "C19-cardiac":
            videos = np.hstack([i for i in videos if i.split(".")[-1] not in video_exts and '-C19-cardiac' in i and os.path.isdir(os.path.join(self.datadir,i))])
        
        #sort videos
        videos.sort()

        print(f"No of videos = {len(videos)}")

        processed_dataset_dict = {}

        for video in videos:

            video_dir = os.path.join(self.datadir, video, self.video_type)

            frame_filenames = [f for f in os.listdir(video_dir) if '.png' in f or '.jpeg' in f or '.jpg' in f or '.JPG' in f]
            frame_filenames.sort()

            
            print(f"Processing video : {video}")

            processed_frame_dict = {}
            for f in frame_filenames:
                processed_frame_dict[f] = os.path.join(video_dir, f) 

            processed_dataset_dict[video] = processed_frame_dict

        return processed_dataset_dict
