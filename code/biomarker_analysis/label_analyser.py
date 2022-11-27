import numpy as np
import matplotlib.pyplot as plt
import json

import os

from sklearn.metrics import cohen_kappa_score

# label_names = [ 
#                     ['al-none', 'al-weak', 'al-bold', 'al-*also* stacked', 'al-*also* wide (> 2cm)'],
#                     ['bl-none', 'bl-few (1-3)', 'bl-some (4-5)', 'bl-many|coalescing', "bl-\"white\" (no striations)"],   
#                     ['pi-none', 'pi-<5mm (single)', 'pi-<5mm (multiple)', 'pi-5-10mm', 'pi->10mm'], 
#                     ['pb-none', 'pb-<5mm (single)', 'pb-<5mm (multiple)', 'pb-5-10mm', 'pb->10mm'],
#                     ['cn-none', 'cn-<5mm (single)', 'cn-<5mm (multiple)', 'cn-5-10mm', 'cn->10mm'],
#                     ['ef-none', 'ef-<5mm (single)', 'ef-<5mm (multiple)', 'ef-5-10mm', 'ef->10mm'],
#                 ]
label_names = [ 
                    ['al-none', 'al-weak', 'al-bold', 'al-stacked', 'al-wide'],
                    ['bl-none', 'bl-few (1-3)', 'bl-some (4-5)', 'bl-many', "bl-white"],   
                    ['pi-none', 'pi-<5 (s)', 'pi-<5 (m)', 'pi-5-10', 'pi->10mm'], 
                    ['pb-none', 'pb-<5 (s)', 'pb-<5 (m)', 'pb-5-10', 'pb->10mm'],
                    ['cn-none', 'cn-<5 (s)', 'cn-<5 (m)', 'cn-5-10', 'cn->10mm'],
                    ['ef-none', 'ef-<5 (s)', 'ef-<5 (m)', 'ef-5-10', 'ef->10mm'],
                ]
label_score = ['score-0', 'score-1', 'score-2', 'score-3']
label_features = ['alines', 'blines', 'pleural_break', 'pleural_indent', 'consolidation', ]

label_features_n_scores = label_features + ['lung-severity']

showFig = False


def calInterRateAgreementScores(user_video_label_data, label_files, video_list):

    assert user_video_label_data.shape == (len(video_list), len(label_files), len(label_features_n_scores))

    stats_file_name = "User_Agreement_Stats.csv"
    
    if os.path.exists(stats_file_name):
        os.remove(stats_file_name)

    f = open(stats_file_name, "a")
    f.write(", , User Agreement Stats, , \n")

    user_names = [i.split('_')[-2] for i in label_files]

    for f_idx, feature in enumerate(label_features_n_scores):

        f.write(", , , , \n\n")
        f.write(f", , {feature}, , \n")
        f.write(", , , , \n")
        f.write(f", {str(user_names).replace('[', '').replace(']', '')}\n")

        user_agreement_matrix = -5*np.ones((len(user_names), len(user_names)))
        for cu_idx, current_user in enumerate(user_names):

            for ru_idx, ref_user in enumerate(user_names):
        
                cu_fdata = np.hstack(user_video_label_data[:, cu_idx, f_idx])
                ru_fdata = np.hstack(user_video_label_data[:, ru_idx, f_idx])

                ck_score = cohen_kappa_score(cu_fdata, ru_fdata)

                #The ck_score for the same combination should match or if yet to be calculated then the default (-5) value should be present
                assert user_agreement_matrix[cu_idx, ru_idx] == -5 or user_agreement_matrix[cu_idx, ru_idx] == ck_score or user_agreement_matrix[ru_idx, cu_idx] == ck_score, "Error: ck_score mismatch!"
                # Cohen score of user with itself should be 1
                assert cu_idx != ru_idx or ck_score == 1, "Error: Cohen score of user with itself not 1!"
                user_agreement_matrix[cu_idx, ru_idx] = ck_score

                print(f"[{feature}] {current_user} vs {ref_user} ck_score = {ck_score}")

            f.write(f"{current_user}, {str([i for i in user_agreement_matrix[cu_idx, :]]).replace('[', '').replace(']', '')}\n")

        print(user_agreement_matrix)

        
        f.write(", , , , \n\n")
    
    f.write(", ***, End of File, ***, \n")

    f.close()


if __name__ == '__main__':

    TRAIN = True #False

    label_root_path = '/home/grg/Desktop/cloudTk/'
    
    label_files = [f for f in os.listdir(label_root_path) if 'user_label_' in f]
    label_files.sort()

    exclude_label_list = ['user_label__1.json', 'user_label_tsmNet_2.json', 'user_label_tsmNet_3.json']
    label_files = [f for f in label_files if f not in exclude_label_list] 

    with open(os.path.join(label_root_path, label_files[0]), 'r') as json_file:
        label_dict = json.load(json_file)
    
    video_list = list(label_dict.keys())

    users_label_dict = {}
    for label_file in label_files:

        with open(os.path.join(label_root_path, label_file), 'r') as json_file:
            label_dict = json.load(json_file)

        users_label_dict[label_file] = label_dict
    
    x_al = np.arange(len(label_names[0]))
    x_bl = np.arange(len(label_names[1]))
    x_pi = np.arange(len(label_names[2]))
    x_pb = np.arange(len(label_names[3]))
    x_cn = np.arange(len(label_names[4]))
    x_ef = np.arange(len(label_names[5]))
    x_ls = np.arange(len(label_score))
    
    x_offset = [(o - len(label_files)/2)/10 for o in range(len(label_files))]

    user_video_label_data = []
    for idx, video in enumerate(video_list):

        fig, axs = plt.subplots(3, 3)

        user_label_data = []
        for l_idx, label_file in enumerate(label_files):

            video_label = users_label_dict[label_file][video]

            # user_label_data.append([video_label[i] for i in label_features])
            user_label_data.append([video_label[i] for i in label_features_n_scores])

            axs[0,0].bar(x_al + x_offset[l_idx], video_label['alines'], label = f"{label_file.split('_')[-2]}", width = 0.1)
            axs[0,1].bar(x_bl + x_offset[l_idx], video_label['blines'], label = f"{label_file.split('_')[-2]}", width = 0.1)
            axs[0,2].bar(x_pi + x_offset[l_idx], video_label['pleural_indent'], label = f"{label_file.split('_')[-2]}", width = 0.1)
            axs[1,0].bar(x_pb + x_offset[l_idx], video_label['pleural_break'], label = f"{label_file.split('_')[-2]}", width = 0.1)
            axs[1,1].bar(x_cn + x_offset[l_idx], video_label['consolidation'], label = f"{label_file.split('_')[-2]}", width = 0.1)
            axs[1,2].bar(x_ef + x_offset[l_idx], video_label['effusion'], label = f"{label_file.split('_')[-2]}", width = 0.1)
            axs[2,1].bar(x_ls + x_offset[l_idx], video_label['lung-severity'], label = f"{label_file.split('_')[-2]}", width = 0.1)
            

            # axs[2,1].bar(x+x_offset[l_idx], lung_sev, label = f"{label_file.split('_')[-2]}", width = 0.1)
            
            # video_label = {}

            # video_label['alines'] = pred[0:5].tolist()

            # video_label['blines'] = pred[5:10].tolist()

            # video_label['pleural_indent'] = pred[10:15].tolist()

            # video_label['pleural_break'] = pred[15:20].tolist()

            # video_label['consolidation'] = pred[20:25].tolist()

            # video_label['effusion'] = [0, 0, 0, 0, 0]

            # video_label['lung-severity'] = [1, 0, 0, 0]

            # video_label['unusual_findings'] = ''

            # label_dict[video] = video_label


        user_video_label_data.append(user_label_data)

        axs[0,0].set_xticks(x_al)
        axs[0,0].set_xticklabels(label_names[0])
        axs[0,0].legend()
        axs[0,0].set_title('A-lines')

        axs[0,1].set_xticks(x_bl)
        axs[0,1].set_xticklabels(label_names[1])
        axs[0,1].legend()
        axs[0,1].set_title('B-lines')

        axs[0,2].set_xticks(x_pi)
        axs[0,2].set_xticklabels(label_names[2])
        axs[0,2].legend()
        axs[0,2].set_title('Pleural Indent')

        axs[1,0].set_xticks(x_pb)
        axs[1,0].set_xticklabels(label_names[3])
        axs[1,0].legend()
        axs[1,0].set_title('Pleural Break')

        axs[1,1].set_xticks(x_cn)
        axs[1,1].set_xticklabels(label_names[4])
        axs[1,1].legend()
        axs[1,1].set_title('Consolidation')

        axs[1,2].set_xticks(x_ef)
        axs[1,2].set_xticklabels(label_names[5])
        axs[1,2].legend()
        axs[1,2].set_title('Effusion')

        axs[2,1].set_xticks(x_ls)
        axs[2,1].set_xticklabels(label_score)
        axs[2,1].legend()
        axs[2,1].set_title('Lung-severity Score')


        fig.set_size_inches(16, 12)

        plt.suptitle(f"Video = {video}")
        plt.savefig(f"Labeling Stats for Video = {video}.png", )
        if showFig:
            plt.show()

    #Save user_video_label_data
    user_video_label_data = np.array(user_video_label_data)
    np.savez("user_video_label_data.npz", user_video_label_data)
    
    calInterRateAgreementScores(user_video_label_data, label_files, video_list)

    print(f"finished!")