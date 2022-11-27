import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.core.defchararray import replace

import torch
from torch.utils.data import Dataset

import cv2
import json

import os

import constants
import utils




class LSULungDataset(Dataset):

    def __init__(self, dataset_type, random_split_trial = 'R1', video_type = 'crop_image', process_data = False, 
                        reprocess = False, augmentData = None, upsampleData = False, foldsToIncludeList = [],
                        shouldAugmentClip = False, clip_width = 18, multiClipVal = False):

        super(LSULungDataset, self).__init__()

        self.datadir = constants.dataset_path
        self.processed_datadir = constants.processed_dataset_path

        
        self.numClasses = 4
        self.labels = np.arange(self.numClasses)
        self.label_names = [ 
                            'score-0', 'score-1', 'score-2', 'score-3',
                            ]
        self.label_features = ['lung-severity', ]

        # self.image_size = (312, 192)
        self.image_size = (224, 224)
        
        # self.clip_sampling_rate = 10
        self.clip_width = clip_width
        # self.clip_width = 36
        # self.clip_width = 150
        print(f"clip_width = {self.clip_width}")

        self.shouldAugmentClip = shouldAugmentClip
        print(f"shouldAugmentClip = {self.shouldAugmentClip}")

        #dataset type
        self.dataset_type = dataset_type 
        
        self.recal_dataset_split = False

        self.random_split_trial = random_split_trial
        self.train_val_test_plit = (0.7, 0.1, 0.2)

        self.video_type = video_type
        print(f"video_type = {self.video_type}")

        self.process_data = process_data

        self.reprocess = reprocess
        self.augmentData = augmentData

        self.processed_dataset_json_path = os.path.join(self.processed_datadir, f"processed_dataset_files_{self.video_type}.json")

        self.label_json_path = os.path.join(self.datadir, "user_label_gautamgare_12.json")
        with open(self.label_json_path, 'r') as json_file:
            self.label_dict = json.load(json_file)

        self.video_names = np.sort(np.array(list(self.label_dict.keys())))

        if self.process_data:
            processed_dataset_dict = self._processData()

            processed_video_names = np.sort(np.array(list(processed_dataset_dict.keys())))
            labeled_video_names = ['.'.join(v.split('.')[:-1]) for v in self.video_names]
            #Check processed videos are all present in label_dict
            assert np.array_equal(labeled_video_names, processed_video_names), "Missing video labels"

        with open(self.processed_dataset_json_path, 'r') as json_file:
            self.processed_dataset_dict = json.load(json_file)

        self.foldsToIncludeList = foldsToIncludeList
        print(f"foldsToIncludeList : {self.foldsToIncludeList}")
        self.videosToIncludeList = self.getVideosFromSplit()
        print(f'videosToIncludeList : {self.videosToIncludeList}')
        
        self.upsampleData = upsampleData
        if self.upsampleData:
            self.__upsampleData()

        self.multiClipVal = multiClipVal
        if self.multiClipVal:
            self.__multiClipsampleData()

        print(f'Using random generated split {self.random_split_trial}')

    
    def __upsampleData(self):

        labels = np.array([np.argmax(self.label_dict[v]['lung-severity']) for v in self.videosToIncludeList])

        assert len(labels) == len(self.videosToIncludeList), "Error! Label count mismatch."
        classes, count = np.unique(labels, return_counts = True)
        
        max_count = max(count)

        label_indices = []
        for c in classes:

            c_idx = np.where(labels == c)[0]
            assert np.unique(labels[c_idx]) == c, "Error! Wrong class index filtered."

            upsample_c_idx = np.random.choice(c_idx, size = max_count, replace = True)

            label_indices.extend(upsample_c_idx)

        assert len(label_indices) == max_count * len(classes)

        self.upsample_label_indices = label_indices

        print(f"Up-sampled classes to {max_count}")

    def __multiClipsampleData(self):

        self.no_multi_clip = 5

        self.multiClip_sample_label_videos = np.array([[f"{v}-sf{f}" for f in range(self.no_multi_clip)] for v in self.videosToIncludeList]).flatten()

        assert len(self.multiClip_sample_label_videos) == len(self.videosToIncludeList)*self.no_multi_clip, "Error! Multi Clip Label count mismatch."

        print(f"multiClip-sampled classes to {len(self.multiClip_sample_label_videos)}")

    def __len__(self):
        if self.upsampleData:
            return len(self.upsample_label_indices)
        if self.multiClipVal:
            return len(self.multiClip_sample_label_videos)
        else:
            return len(self.videosToIncludeList)

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

        frames_dict = self.processed_dataset_dict[video_key]
        frames_list = list(frames_dict.keys())

        # clip_stride = np.round(len(frames_list)/self.clip_width)
        clip_stride = np.floor(len(frames_list)/self.clip_width)

        if self.multiClipVal:
            start_frame = np.arange(0, clip_stride, int(clip_stride/self.no_multi_clip))[sf_idx]
        else:
            start_frame = np.random.randint(clip_stride)
        
        # clip_frame_ids = np.arange(0, len(frames_list), self.clip_sampling_rate)
        # clip_frame_ids = np.arange(0, len(frames_list), np.round(len(frames_list)/self.clip_width)).astype(np.int)
        clip_frame_ids = np.arange(start_frame, len(frames_list), clip_stride).astype(np.int)

        #Slice the clips
        clip_frame_ids = clip_frame_ids[:self.clip_width]

        assert len(clip_frame_ids) == self.clip_width, "Clip width mismatch"

        clip = torch.Tensor([torch.load(frames_dict[frames_list[f]]) for f in clip_frame_ids]) #Dim - TxHxWxC

        #Convert image from 0-255 to 0-1 range
        clip = clip/255.0


        if self.shouldAugmentClip:
            clip = self.augmentClip(clip)

        video_label = self.label_dict[video_name]

        label = []
        [label.extend(video_label[f]) for f in self.label_features]
        if self.biomarkerLabels:
            label = torch.tensor(label).float()
        elif self.NIH_scores:
            label = torch.tensor(self.severityScore_dict[video_name.split('(')[1][-1]])
        elif self.NIH_scores:
            label = torch.tensor(self.severityScore_dict[video_name.split('(')[1][-1]])
        else:
            label = torch.tensor(np.argmax(label))

        ccl_label = torch.tensor([0])

        rf = torch.zeros(1)
            
        return video_name, clip, rf, label, ccl_label


    def augmentClip(self, clip):

        probHFlip = np.random.random()

        if probHFlip > 0.5:
            clip = torch.flip(clip, [-1])

        
        # probVFlip = np.random.random()

        # if probVFlip > 0.5:
        #     clip = torch.flip(clip, [-2])

        probIntensity = np.random.random()

        if probIntensity > 0.5:
            scale = np.random.choice([0.8, 0.9, 1.1], size = 1)[0]
            clip = clip * scale
            clip = torch.clamp(clip, min=0, max=1)

        return clip

    def getAnalysisSample_v0(self):

        analysis_sample_path = os.path.join(self.datadir, f'analysis_sample_{self.random_split_trial}.npz')

        if not os.path.exists(analysis_sample_path):  

           analysis_sample = self.__getitem__(np.random.randint(low = 0, high = self.__len__()))     

           np.savez(analysis_sample_path, analysis_sample) 

        # analysis_sample = np.load(analysis_sample_path, allow_pickle = True).get('arr_0')         

        # return analysis_sample

        filename_array, grey_array, rf_array, label_array, ccl_label_array = np.load(analysis_sample_path, allow_pickle = True).get('arr_0')

        filename_array = [f.replace('crop_image', '').replace('linear_probe_straighten_pleural_line', '').replace('linear_probe_under_pleural_line', '')   for f in filename_array]

        grey_array = []
        for idx, f in enumerate(filename_array):

            video, frame = f.split('__')

            frame_path = os.path.join(self.greys_dir, self.label_names[label_array[idx]], video, self.video_type, frame)
            if not os.path.exists(frame_path):
                
                if 'frame' in frame:
                    frame = frame.replace('frame', '') 
                    frame = f"{int(frame.split('.')[0]):05d}.pt"
                elif frame.split('.')[0].isnumeric():
                    frame = f"frame{int(frame.split('.')[0])}.pt"

                frame_path = os.path.join(self.greys_dir, self.label_names[label_array[idx]], video, self.video_type, frame)

                assert os.path.exists(frame_path), "file not found!"

            grey_array.append(torch.load(frame_path))

        
        # grey_array = [i.unsqueeze(0) for i in grey_array]
        rf_array = [rf_array for i in range(len(filename_array))]

        return filename_array, grey_array, rf_array, label_array, ccl_label_array

    def getAnalysisSample(self):

        
        filename_array = []
        grey_array = []
        rf_array = []
        label_array = []
        ccl_label_array = []
        
        for idx in range(self.__len__()):

            video_name, clip, rf, label, ccl_label = self.__getitem__(idx)

            filename_array.append(video_name)
            grey_array.append(clip)
            rf_array.append(rf)
            label_array.append(label)
            ccl_label_array.append(ccl_label)
        
        return filename_array, grey_array, rf_array, label_array, ccl_label_array

    def getVideosFromSplit(self):
        
        split_json_path = os.path.join(self.datadir, f'dataset_split_equi_class_{self.random_split_trial}.json')

        if not os.path.exists(split_json_path) or self.recal_dataset_split:
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

    def generateDataSplitEquiClass_v0(self, split_json_path):
        
        print(f"Generating data split : {split_json_path}")

        train_per, val_per, test_per = self.train_val_test_plit

        with open(self.processed_dataset_json_path, 'r') as json_file:
            processed_dataset_dict = json.load(json_file)

        train_video_dict = {}
        test_video_dict = {}
        val_video_dict = {}

        lung_severity = np.array([np.argmax(self.label_dict[l]['lung-severity']) for l in self.video_names])
        
        #Process all severity images
        for severity in np.unique(lung_severity):
            video_idx = np.where(lung_severity == severity)

            video_list = self.video_names[video_idx]
            
            assert np.unique([np.argmax(self.label_dict[l]['lung-severity']) for l in video_list]) == severity

            np.random.shuffle(video_list)

            n = len(video_list)

            no_test_videos = max(int(np.round(n*test_per)), 1)
            no_val_videos = max(int(np.round(n*val_per)), 1)
            # no_val_videos = max(int(np.round(n*val_per)), 1)
            no_train_videos = n - no_test_videos - no_val_videos

            print(f"{severity} : Total no of Videos = {n}; [Train = {no_train_videos}; Test = {no_test_videos}; Val = {no_val_videos}")

            test_video_dict[str(severity)] = video_list[: no_test_videos].tolist()
            val_video_dict[str(severity)] = video_list[no_test_videos : no_test_videos + no_val_videos].tolist()
            train_video_dict[str(severity)] = video_list[no_test_videos + no_val_videos :].tolist()

            assert no_val_videos == len(val_video_dict[str(severity)])
            assert no_test_videos == len(test_video_dict[str(severity)])
            assert no_train_videos == len(train_video_dict[str(severity)])

        split_dict = {"train": train_video_dict, "val": val_video_dict, "test": test_video_dict}

        #Save splits
        split_json = json.dumps(split_dict, indent=4)
        f = open(split_json_path,"w")
        f.write(split_json)
        f.close()


    def generateDataSplitEquiClass_v1(self, split_json_path):
        
        print(f"Generating data split : {split_json_path}")

        train_per, val_per, test_per = self.train_val_test_plit

        with open(self.processed_dataset_json_path, 'r') as json_file:
            processed_dataset_dict = json.load(json_file)

        train_video_dict = {}
        test_video_dict = {}
        val_video_dict = {}

        lung_severity = np.array([np.argmax(self.label_dict[l]['lung-severity']) for l in self.video_names])

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
            
            assert np.unique([np.argmax(self.label_dict[l]['lung-severity']) for l in video_list]) == severity

            np.random.shuffle(video_list)

            n = len(video_list)

            no_test_videos = max(int(np.round(n*test_per)), 1)

            #Make test videos count even
            no_test_videos += 1 if no_test_videos % 2 == 1 else 0

            no_val_videos = max(int(np.round(n*val_per)), 1)
            # no_val_videos = max(int(np.round(n*val_per)), 1)
            no_train_videos = n - no_test_videos - no_val_videos

            print(f"{severity} : Total no of Videos = {n}; [Train = {no_train_videos}; Val = {no_val_videos}; Test = {no_test_videos}]")

            #Select Test video's from unique severity pt list:
            unique_sev_video_list = [v for v in video_list if v.split('(')[0] in pts_with_same_sev]
            assert len(unique_sev_video_list) >= no_test_videos, "Error! Videos for unique severity is less than needed for test set."
            
            unique_sev_pts_list = np.unique([v.split('(')[0] for v in unique_sev_video_list])

            sel_test_pts = np.random.choice(unique_sev_pts_list, size = int(no_test_videos/2), replace = False) #Note: For now, we have 2 videos per patients
            test_videos = [v for v in unique_sev_video_list if v.split('(')[0] in sel_test_pts]

            test_video_dict[str(severity)] = test_videos

            non_test_videos = [v for v in video_list if v not in test_video_dict[str(severity)]]
            
            val_video_dict[str(severity)] = non_test_videos[: no_val_videos]
            train_video_dict[str(severity)] = non_test_videos[no_val_videos :]

            assert no_val_videos == len(val_video_dict[str(severity)])
            assert no_test_videos == len(test_video_dict[str(severity)])
            assert no_train_videos == len(train_video_dict[str(severity)])


        test_pts = np.unique(np.hstack([[v.split('(')[0] for v in v_list] for v_list in test_video_dict.values()]))
        val_pts = np.unique(np.hstack([[v.split('(')[0] for v in v_list] for v_list in val_video_dict.values()]))
        train_pts = np.unique(np.hstack([[v.split('(')[0] for v in v_list] for v_list in train_video_dict.values()]))

        for t_pt in test_pts:
            assert t_pt not in val_pts, f"Error! test patient {t_pt} is in val set"
            assert t_pt not in train_pts, f"Error! test patient {t_pt} is in train set"

        split_dict = {"train": train_video_dict, "val": val_video_dict, "test": test_video_dict}

        #Save splits
        split_json = json.dumps(split_dict, indent=4)
        f = open(split_json_path,"w")
        f.write(split_json)
        f.close()

    
    def generateDataSplitEquiClass(self, split_json_path):
        
        print(f"Generating data split : {split_json_path}")

        train_per, val_per, test_per = self.train_val_test_plit

        folds = 5
        folds_list = ["A", "B", "C", "D", "E"]

        with open(self.processed_dataset_json_path, 'r') as json_file:
            processed_dataset_dict = json.load(json_file)

        train_video_dict = {}
        # val_video_dict = {}
        test_video_dict = {}

        lung_severity = np.array([np.argmax(self.label_dict[l]['lung-severity']) for l in self.video_names])

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
            
            assert np.unique([np.argmax(self.label_dict[l]['lung-severity']) for l in video_list]) == severity

            np.random.shuffle(video_list)

            n = len(video_list)

            no_test_videos = max(int(np.round(n*test_per)), 1)

            #Make test videos count even
            no_test_videos += 1 if no_test_videos % 2 == 1 else 0

            # no_val_videos = max(int(np.round(n*val_per)), 1)
            # no_val_videos = max(int(np.round(n*val_per)), 1)
            # no_train_videos = n - no_test_videos - no_val_videos
            
            no_train_videos = n - no_test_videos

            print(f"{severity} : Total no of Videos = {n}; [Train = {no_train_videos}; Test = {no_test_videos}]")

            #Select Test video's from unique severity pt list:
            unique_sev_video_list = [v for v in video_list if v.split('(')[0] in pts_with_same_sev]
            assert len(unique_sev_video_list) >= no_test_videos, "Error! Videos for unique severity is less than needed for test set."
            
            unique_sev_pts_list = np.unique([v.split('(')[0] for v in unique_sev_video_list])

            sel_test_pts = np.random.choice(unique_sev_pts_list, size = int(no_test_videos/2), replace = False) #Note: For now, we have 2 videos per patients
            test_videos = [v for v in unique_sev_video_list if v.split('(')[0] in sel_test_pts]

            test_video_dict[str(severity)] = test_videos

            # val_video_dict[str(severity)] = non_test_videos[: no_val_videos]
            # train_video_dict[str(severity)] = non_test_videos[no_val_videos :]

            non_test_videos = [v for v in video_list if v not in test_video_dict[str(severity)]]
            np.random.shuffle(non_test_videos)

            assert len(non_test_videos) == no_train_videos, "Error! Train video count mismatch."
        
            fold_len = int(np.floor(no_train_videos/folds))

            train_folds_dict = {}
            for fold_id in range(folds):
                train_folds_dict[folds_list[fold_id]] = non_test_videos[fold_len*fold_id:fold_len*(fold_id+1)]
            
            #Append the remaining vodeos 
            f_c = 0
            for video in non_test_videos[fold_len*(fold_id+1):]:
                train_folds_dict[folds_list[f_c]].append(video)
                f_c+=1
            
            train_video_dict[str(severity)] = train_folds_dict

            # assert no_val_videos == len(val_video_dict[str(severity)])
            assert no_test_videos == len(test_video_dict[str(severity)])
            assert no_train_videos == sum([len(l) for l in train_video_dict[str(severity)].values()])

            print(f"{severity} : Train-folds - {[len(l) for l in train_video_dict[str(severity)].values()]}")


        test_pts = np.unique(np.hstack([[v.split('(')[0] for v in v_list] for v_list in test_video_dict.values()]))
        # val_pts = np.unique(np.hstack([[v.split('(')[0] for v in v_list] for v_list in val_video_dict.values()]))
        train_pts = []
        [[train_pts.extend([v.split('(')[0] for v in v_list]) for v_list in l.values()] for l in train_video_dict.values()]
        train_pts = np.unique(train_pts)

        for t_pt in test_pts:
            # assert t_pt not in val_pts, f"Error! test patient {t_pt} is in val set"
            assert t_pt not in train_pts, f"Error! test patient {t_pt} is in train set"

        split_dict = {"train": train_video_dict, "test": test_video_dict}

        #Save splits
        split_json = json.dumps(split_dict, indent=4)
        f = open(split_json_path,"w")
        f.write(split_json)
        f.close()



    def _processData(self):

        #Remove previous preprocessed data and create new
        if self.reprocess:
            utils.createDir(os.path.join(self.processed_datadir))
            print(f'Re-processing input for {self.datadir}')


        video_data_dir = os.path.join(self.datadir, 'videos')
        videos = [f for f in os.listdir(video_data_dir) if os.path.isdir(os.path.join(video_data_dir, f))]
        print(f"No of videos = {len(videos)}")

        processed_dataset_dict = {}
        if os.path.exists(self.processed_dataset_json_path):
            
            with open(self.processed_dataset_json_path, 'r') as json_file:
                processed_dataset_dict = json.load(json_file)

        for video in videos:

            video_dir = os.path.join(video_data_dir, video, self.video_type)

            frame_filenames = [f for f in os.listdir(video_dir) if '.png' in f or '.jpeg' in f or '.jpg' in f or '.JPG' in f]
            frame_filenames.sort()

            if video in processed_dataset_dict:

                processed_frames = list(processed_dataset_dict[video].keys())
                processed_frames.sort()

                if np.array_equal(frame_filenames, processed_frames):
                    continue
            
            self.__processFrames(processed_dataset_dict, video, video_dir, frame_filenames)



        #Save processed dataset dict
        processed_dataset_json = json.dumps(processed_dataset_dict, indent=4)
        f = open(self.processed_dataset_json_path, "w")
        f.write(processed_dataset_json)
        f.close()

        return processed_dataset_dict

    def __processFrames(self, processed_dataset_dict, video, video_dir, frame_filenames):

        print(f"Processing video : {video}")

        #Set Flag to recalculate dataset split
        self.recal_dataset_split = True

        #Create processed Video dir
        processed_video_dir = os.path.join(self.processed_datadir, video, self.video_type)
        utils.createDir(processed_video_dir, exist_ok = True)


        processed_frame_dict = {}
        for frame_filename in frame_filenames:       

            grey_path = os.path.join(video_dir, frame_filename)

            # processed_grey_path = os.path.join(processed_video_dir, frame_filename)
            
            grey = cv2.imread(grey_path)
            
            org_filename = '.'.join(frame_filename.split('.')[:-1])

            if grey.shape[:-1] != self.image_size:
                grey = cv2.resize(grey, (self.image_size[1], self.image_size[0]))

            assert grey.shape[:-1] == self.image_size

            grey = grey[:, :, 0][np.newaxis]
            

            if self.augmentData is None:
                torch.save(grey, os.path.join(processed_video_dir, org_filename +'.pt'))
                matplotlib.image.imsave(os.path.join(processed_video_dir, org_filename+'.png'), grey.squeeze().data)

                #Change frame filename by only retaining frame1.png from video_name_frame1.png
                if len(frame_filename.split("_")) > 1:
                    frame_filename = frame_filename.split("_")[-1]

                processed_frame_dict[frame_filename] = os.path.join(processed_video_dir, org_filename +'.pt')
        
        processed_dataset_dict[video] = processed_frame_dict


        

