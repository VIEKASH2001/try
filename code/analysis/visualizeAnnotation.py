"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-03 21:20:56
 * @modify date 2021-11-26 19:36:51
 * @desc [description]
 """
 
import cv2
import matplotlib.pyplot as plt

import numpy as np 

import os

import imageio

import utils
        

        

def savePredAsGIF(clip, target, pred, path, filename):

    # input = input.permute(0, 1, 3, 4, 2).numpy()

    txt_ht = 80

    (img_height, img_width,) = clip.shape[-3:-1]

    
        
    image_list = []
    for frame_no, frame in enumerate(clip):

        # frame = np.tile(frame, (1,1,3))

        # #Convert img from BGR to RGB color format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame = (frame*255.0).astype(np.uint8)

        result_img = getTextImg(txt_ht, img_width, frame_no, target, pred, filename)

        frame = np.concatenate((result_img, frame), axis = 0)
        # plt.imshow(frame)
        # plt.show()
        image_list.append(frame)

    saveAsVideo(image_list, path)

    return image_list


def saveSegPredAsGIF(clip, seg_target, seg_pred, target, pred, path, filename):

    txt_ht = 80

    (img_height, img_width,) = clip.shape[-3:-1]

    
        
    image_list = []
    for frame_no, (clip_f, seg_target_f, seg_pred_f) in enumerate(zip(clip, seg_target, seg_pred)):

        # frame = np.tile(frame, (1,1,3))

        # #Convert img from BGR to RGB color format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame = (frame*255.0).astype(np.uint8)


        result_img = getTextImg(txt_ht, img_width*3, frame_no, target, pred, filename)

        r1 = np.concatenate((clip_f, seg_target_f, seg_pred_f), axis = 1)

        frame = np.concatenate((result_img, r1), axis = 0)
        # plt.imshow(frame)
        # plt.show()
        image_list.append(frame)

    saveAsVideo(image_list, path)

    return image_list


def saveGradCAMPredAsGIF(clip, cam, gb, cam_gb, target, pred, path, filename):

    # # input = input.permute(0, 1, 3, 4, 2).numpy()
    # clip = clip.squeeze()[:, :, :, np.newaxis]

    txt_ht = 80

    (img_height, img_width) = clip.shape[-3:-1]

 
    image_list = []
    for frame_no, (clip_f, cam_f, gb_f, cam_gb_f) in enumerate(zip(clip, cam, gb, cam_gb)):

        # clip_f = np.tile(clip_f, (1,1,3))

        # gb_f = np.tile(gb_f, (1,1,3))

        # #Convert img from BGR to RGB color format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # clip_f = (clip_f*255.0).astype(np.uint8)


        result_img = getTextImg(txt_ht, img_width*2, frame_no, target, pred, filename)

        r1 = np.concatenate((clip_f, gb_f), axis = 1)
        r2 = np.concatenate((cam_f, cam_gb_f), axis = 1)

        frame = np.concatenate((result_img, r1, r2), axis = 0)
        # plt.imshow(frame)
        # plt.show()
        image_list.append(frame)

    saveAsVideo(image_list, path)

    return image_list



class VideoPredWriter():

    def __init__(self, video_path, fps = 15):

        self.writer = imageio.get_writer(video_path, fps = fps)

        self.frame_idx = 0

        self.txt_ht = 40


    def addFrame(self, frame, pred):

        self.frame_idx += 1

        (img_height, img_width,) = frame.shape[-2:]
        
        frame = frame.squeeze().permute(1,2,0).numpy()
        #Convert img from BGR to RGB color format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = (frame*255.0).astype(np.uint8)

        result_img = getTextImg(self.txt_ht, img_width, frame_no = self.frame_idx, predicted_activities = pred)

        frame = np.concatenate((result_img, frame), axis = 0)

        self.writer.append_data(frame)

            
    def close(self):

        self.writer.close()


def getTextImg(txt_ht, img_width, frame_no = None, target_activities = None, predicted_activities = None, video_name = None):

    font_scale = 0.4
    y_place = 5
    x_place = 12

    result_img = np.zeros((txt_ht, img_width, 3), np.uint8)
    
    #Add video name
    if video_name is not None:
        cv2.putText(result_img, f"{video_name}",
                    (y_place,x_place), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (0, 255, 255), 
                    1)

        x_place += 20
    
    #Add frame no and avg IoU
    if frame_no is not None:
        cv2.putText(result_img, f"Frame No.: {frame_no}",
                (y_place,x_place), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 255), 
                1)

        x_place += 20
    
    
    #Add frame no and avg IoU
    if target_activities is not None:
        cv2.putText(result_img, f"Target: {target_activities}",
                (y_place,x_place), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 255), 
                1)

        x_place += 20
    
    #Add frame no and avg IoU
    if predicted_activities is not None:
        cv2.putText(result_img, f"Prediction: {predicted_activities}",
                (y_place,x_place),  
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 255, 255), 
                1)

    return result_img




def saveInputArrayAsGIF(input, path):

    input = input.permute(0, 1, 3, 4, 2).numpy()
    
    for idx, frames in enumerate(input):

        saveInputAsGIF(frames, os.path.join(path, f"input{idx}.gif"))


def saveInputAsGIF(frames, path):

    image_list = []
    for frame in frames:
        #Convert img from BGR to RGB color format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = (frame*255.0).astype(np.uint8)

        # plt.imshow(frame)
        # plt.show()
        image_list.append(frame)

    saveAsVideo(image_list, path)





def saveAsVideo(image_list, video_name='temp.gif'):
    # # out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps = 15, frameSize = len(image_list))
    # out = cv2.VideoWriter(filename = video_name, apiPreference = cv2.CAP_FFMPEG,  fourcc = cv2.VideoWriter_fourcc(*'DIVX'), fps = 1, frameSize = image_list[0].shape[:2])
    # for img in image_list:
    #     out.write(img)
    # out.release()
    imageio.mimsave(video_name, image_list)