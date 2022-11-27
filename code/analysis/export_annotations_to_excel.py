"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2022-01-07 21:35:35
 * @modify date 2022-01-07 22:43:40
 * @desc [description]
 """
 

from re import A
import numpy as np

import os

import json

import xlsxwriter

def readJson(json_path):

    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)

    return json_dict


def readFile(file_path):

    with open(file_path, 'r') as f:
        # file = f.read()
        file = f.readlines()

    return file

def excelInsertColumn(worksheet, col, row, header, items):

    worksheet.write(row, col, header)

    # Iterate over the data and write it out row by row.
    [worksheet.write(r, col, i) for r, i in zip(row + 1 + np.arange(len(items)), items)]


def addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name, feature):

    worksheet = workbook.add_worksheet(sheet_name)
    

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 1

    excelInsertColumn(worksheet, col, row, "Videos", xr_video_list)

    # for file in annotation_files:
    for user, id in xr_user_dict.items():
        
        if user in xr_exclude_users:
            continue

        col += 1
        
        annotation = readJson(os.path.join(annotation_path, [i for i in annotation_files if id in i][0]))


        assert len(annotation.keys()) >= len(xr_video_list), "Error! Missing annotation."


        def getSeverityScore(annotation, feature):

            severity_scores = np.array(list(annotation.values())[:-1])[feature]

            return severity_scores

        scores = [getSeverityScore(annotation[video], feature) for video in xr_video_list]

        excelInsertColumn(worksheet, col, row, user, scores)


def getBrixiaScore(i, a):
            score = -1

            if a == 0 and i == 0:
                score = 0
            elif i > 0 and a == 0:
                score = 1
            elif i > 0 and a > 0 and i > a:
                score = 2
            elif i > 0 and a > 0 and a > i:
                score = 3
            elif i == 0 and a > 0:
                score = 3
            elif i > 0 and a > 0 and i == a:
                score = 3
            else:
                raise Exception(f"Error! Undefined logic for a = {a} and i = {i}")
            
            return score


def addBrixiaFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name, feature):

    worksheet = workbook.add_worksheet(sheet_name)
    

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 1

    excelInsertColumn(worksheet, col, row, "Videos", xr_video_list)

    # for file in annotation_files:
    for user, id in xr_user_dict.items():
        
        if user in xr_exclude_users:
            continue

        col += 1
        
        annotation = readJson(os.path.join(annotation_path, [i for i in annotation_files if id in i][0]))


        assert len(annotation.keys()) >= len(xr_video_list), "Error! Missing annotation."


        def getSeverityScore(annotation, feature):

            severity_scores = np.array(list(annotation.values())[:-1])

            brixia_scores = getBrixiaScore(severity_scores[feature[0]].item(), severity_scores[feature[1]].item())

            return brixia_scores

        scores = [getSeverityScore(annotation[video], feature) for video in xr_video_list]

        excelInsertColumn(worksheet, col, row, user, scores)


if __name__ == "__main__":

    annotation_path = "/data1/datasets/LSU_annotations"

    annotation_files = [f for f in os.listdir(annotation_path)]

    us_user_dict = readJson(os.path.join(annotation_path, "user_dict.json"))
    xr_user_dict = readJson(os.path.join(annotation_path, "xray_user_dict.json"))

    us_exclude_users = ["ggare@andrew.cmu.edu", "", "thf214@gmaill.com", "plowe1@lsuhsc.eu", "kzamor", "htran5@lsuhsc.edu", "bdebois@lsuhsc.edu", "bdebois"]
    xr_exclude_users = ["ggare@andrew.cmu.edu", "", "thf214@gmail.com", "bdeboi@lsuhsc.edu", "DSMI18@LSUHSC.EDU"]

    us_video_list = readFile(os.path.join(annotation_path, "videoFiles.txt"))
    xr_video_list = readFile(os.path.join(annotation_path, "xray_videoFiles.txt"))

    us_video_list = [video.split(":")[1][1:-1]for video in us_video_list]
    xr_video_list = [video.split(":")[1][1:-1]for video in xr_video_list]

    #Correct file name "179/178-3.avi" to "179/179-3.avi"
    us_video_list[us_video_list.index("179/178-3.avi")] = "179/179-3.avi"

    # us_video_list = [v for v in us_video_list if v not in ["153", "154", "155", "152", "166b"]]
    xr_video_list = [v for v in xr_video_list if v not in ["153", "154", "155"]]

    us_video_list.sort(key = lambda x: (int(x.split("/")[0]), x.split("/")[1].replace("-", "")))
    # xr_video_list.sort(key = lambda x: (int(x.split("/")[0]), x.split("/")[1].replace("-", "")))


    workbook = xlsxwriter.Workbook('LSU_Consolidated_annotations.xlsx')
    
    # By default worksheet names in the spreadsheet will be
    # Sheet1, Sheet2 etc., but we can also specify a name.
    worksheet = workbook.add_worksheet("Ultrasound (severity)")
    

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 1

    excelInsertColumn(worksheet, col, row, "Videos", us_video_list)

    # for file in annotation_files:
    for user, id in us_user_dict.items():
        
        if user in us_exclude_users:
            continue

        col += 1
        
        annotation = readJson(os.path.join(annotation_path, [i for i in annotation_files if id in i][0]))

        #Correct file name "179/178-3.avi" to "179/179-3.avi"
        if "179/178-3.avi" in annotation:
            annotation["179/179-3.avi"] = annotation["179/178-3.avi"]

        assert len(annotation.keys()) >= len(us_video_list), "Error! Missing annotation."

        scores = [np.argmax(annotation[video]["lung-severity"]) for video in us_video_list]

        excelInsertColumn(worksheet, col, row, user, scores)

     
    worksheet = workbook.add_worksheet("X-Ray (mean severity)")
    

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 1

    excelInsertColumn(worksheet, col, row, "Videos", xr_video_list)

    # for file in annotation_files:
    for user, id in xr_user_dict.items():
        
        if user in xr_exclude_users:
            continue

        col += 1
        
        annotation = readJson(os.path.join(annotation_path, [i for i in annotation_files if id in i][0]))


        assert len(annotation.keys()) >= len(xr_video_list), "Error! Missing annotation."


        def getAvgScore(annotation):

            severity_scores = np.array(list(annotation.values())[:-1])

            mean_severity = severity_scores.mean()

            return mean_severity

        scores = [getAvgScore(annotation[video]) for video in xr_video_list]

        excelInsertColumn(worksheet, col, row, user, scores)

    
    worksheet = workbook.add_worksheet("X-Ray (mean brixia scores)")
    

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 1

    excelInsertColumn(worksheet, col, row, "Videos", xr_video_list)

    # for file in annotation_files:
    for user, id in xr_user_dict.items():
        
        if user in xr_exclude_users:
            continue

        col += 1
        
        annotation = readJson(os.path.join(annotation_path, [i for i in annotation_files if id in i][0]))


        assert len(annotation.keys()) >= len(xr_video_list), "Error! Missing annotation."
        

        def getAvgScore(annotation):

            severity_scores = np.array(list(annotation.values())[:-1])

            indexs = np.array([
                                [(0, 0), (1, 0)],
                                [(0, 1), (1, 1)],
                                [(2, 0), (3, 0)],
                                [(2, 1), (3, 1)],
                                [(4, 0), (5, 0)],
                                [(4, 1), (5, 1)],
                            ])

            brixia_scores = np.array([getBrixiaScore(severity_scores[tuple(x[0])].item(), severity_scores[(tuple(x[1]))].item()) for x in indexs])

            mean_severity = brixia_scores.mean()

            return mean_severity

        scores = [getAvgScore(annotation[video]) for video in xr_video_list]

        excelInsertColumn(worksheet, col, row, user, scores)
    

    worksheet = workbook.add_worksheet("X-Ray (mean interstitial)")
    

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 1

    excelInsertColumn(worksheet, col, row, "Videos", xr_video_list)

    # for file in annotation_files:
    for user, id in xr_user_dict.items():
        
        if user in xr_exclude_users:
            continue

        col += 1
        
        annotation = readJson(os.path.join(annotation_path, [i for i in annotation_files if id in i][0]))


        assert len(annotation.keys()) >= len(xr_video_list), "Error! Missing annotation."


        def getAvgItrScore(annotation):

            severity_scores = np.array(list(annotation.values())[:-1])[::2]

            mean_severity = severity_scores.mean()

            return mean_severity

        scores = [getAvgItrScore(annotation[video]) for video in xr_video_list]

        excelInsertColumn(worksheet, col, row, user, scores)



    worksheet = workbook.add_worksheet("X-Ray (mean alveolar)")
    

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 1

    excelInsertColumn(worksheet, col, row, "Videos", xr_video_list)

    # for file in annotation_files:
    for user, id in xr_user_dict.items():
        
        if user in xr_exclude_users:
            continue

        col += 1
        
        annotation = readJson(os.path.join(annotation_path, [i for i in annotation_files if id in i][0]))


        assert len(annotation.keys()) >= len(xr_video_list), "Error! Missing annotation."


        def getAvgAlvScore(annotation):

            severity_scores = np.array(list(annotation.values())[:-1])[1::2]

            mean_severity = severity_scores.mean()

            return mean_severity

        scores = [getAvgAlvScore(annotation[video]) for video in xr_video_list]

        excelInsertColumn(worksheet, col, row, user, scores)

    

    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (R1 interstitial)", feature = (0,0))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (L1 interstitial)", feature = (0,1))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (R1 alveolar)", feature = (1,0))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (L1 alveolar)", feature = (1,1))

    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (R2 interstitial)", feature = (2,0))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (L2 interstitial)", feature = (2,1))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (R2 alveolar)", feature = (3,0))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (L2 alveolar)", feature = (3,1))

    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (R3 interstitial)", feature = (4,0))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (L3 interstitial)", feature = (4,1))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (R3 alveolar)", feature = (5,0))
    addFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = "X-Ray (L3 alveolar)", feature = (5,1))
    

    region_indexs_dict = {
                            "R1":    [(0, 0), (1, 0)],
                            "L1":    [(0, 1), (1, 1)],
                            "R2":    [(2, 0), (3, 0)],
                            "L2":    [(2, 1), (3, 1)],
                            "R3":    [(4, 0), (5, 0)],
                            "L3":    [(4, 1), (5, 1)],
                        }

    for region, index in region_indexs_dict.items():
        addBrixiaFeatureSheet(worksheet, xr_video_list, xr_user_dict, xr_exclude_users, annotation_path, annotation_files, sheet_name = f"X-Ray ({region} brixia)", feature = index)
   
    
    workbook.close()