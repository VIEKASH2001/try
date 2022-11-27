import numpy as np
import matplotlib.pyplot as plt

import os

import cv2


def convert_video_to_frames(video_name):

    # Initialize the frame number and create empty frame list
    cam = cv2.VideoCapture(video_name)
    frame_num = 0
    frame_list = []

    # Loop until there are frames left
    while True:
        try:
            # Try to read a frame. Okay is a BOOL if there are frames or not
            okay, frame = cam.read()

            # Break if there are no other frames to read
            if not okay:
                break
            # Append to empty frame list
            frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            # print(frame.shape)
            # Increment value of the frame number by 1
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break
    print(len(frame_list))

    return np.array(frame_list)



def readVideosToProcess(rootDir):

    video_exts = ["avi", "mp4", "gif", "mov", "mpeg"]

    videoFiles = np.hstack([[os.path.join(f,i) for i in os.listdir(os.path.join(rootDir,f)) if i.split(".")[-1] in video_exts] for f in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,f))])

    #Filter to retain only convex probe (C19) videos
    videoFiles = [v for v in videoFiles if '-C19-' in v]

    no_of_videos = len(videoFiles)

    #sort videos
    videoFiles.sort()

    assert no_of_videos > 0, "No videos found!"

    return videoFiles



def cropConvexVideos(rootDir):

    videoFiles = readVideosToProcess(rootDir)

    for video in videoFiles:

        # video_name = video.split('/')[-1]
        video_name = ".".join(video.split(".")[:-1])

        video_path = os.path.join(rootDir, video)

        framesList = convert_video_to_frames(video_path)

        if False:
            frame = framesList[0]
            plt.imshow(frame)
            plt.show()
        
        processed_framesList = framesList.copy()

        #Remove right-side Sonosite Logo
        x_low, y_low, x_high, y_high = (660, 93, 689, 120)
        processed_framesList[:, y_low:y_high+1, x_low:x_high+1] = 0


        #Remove left-side Sonosite Logo
        x_low, y_low, x_high, y_high = (420, 93, 448, 120)
        processed_framesList[:, y_low:y_high+1, x_low:x_high+1] = 0

        #Crop the frame
        # x_low, y_low, x_high, y_high = (155, 93, 940, 615)
        # x_low, y_low, x_high, y_high = (155, 93, 940, 508)
        x_low, y_low, x_high, y_high = (155, 93, 940, 602)
        processed_framesList = processed_framesList[:, y_low:y_high+1, x_low:x_high+1]

        if True:
            # frame = processed_framesList[0]
            # plt.imshow(frame)
            # plt.show()

            fig, ax = plt.subplots(1,2)
            ax[0].imshow(framesList[0])
            ax[1].imshow(processed_framesList[0])
            plt.title(video_name)
            plt.show()

        saveFrames(rootDir, video_name, processed_framesList)


def saveFrames(rootDir, video, framesList, dir_name = 'crop_image'):

    dir_path = os.path.join(rootDir, video, dir_name)

    os.makedirs(dir_path)

    for i, f in enumerate(framesList):

        cv2.imwrite(os.path.join(dir_path, f"{i:05d}.png"), f)

if __name__ == "__main__":

    rootDir = "/home/grg/Research/LungUS-AI/data/LSU_Linear_vs_Convex_Dataset"

    cropConvexVideos(rootDir)