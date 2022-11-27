import numpy as np

if __name__ == '__main__':

    path = '/home/grg/OCT_Ultrasound_Segmentation/results/PatientSeverity_Exp/resnet50_Lung-Patient-Severity-Dataset_cntLR_pleuralSeg_T1_R1/Pt1_Prob_based_Eval_Clasification_results.csv'

    y_prob_path = '/home/grg/OCT_Ultrasound_Segmentation/results/PatientSeverity_Exp/resnet50_Lung-Patient-Severity-Dataset_cntLR_pleuralSeg_T1_R1/Pt1_Prob_based_Eval_Clasification_y_prob.npy'



    # convert = lambda x: np.array(list(x))
    # data = np.loadtxt(path, converters={0:np.str, 1:np.float, 2:convert}, delimiter = ";", dtype = np.object)
    data = np.loadtxt(path, delimiter = ";", dtype = np.object)
    y_prob = np.load(y_prob_path)

    filenames_list = np.array([f.replace('crop_image', '').replace('linear_probe_straighten_pleural_line', '').replace('linear_probe_under_pleural_line', '').split('__')[0] for f in data[:, 0]])
    filenames = np.unique(filenames_list)

    avg_softmax_pred_dict = {}
    for filename in filenames:

        file_idx = np.where(filenames_list == filename)

        avg_softmax_pred = y_prob[file_idx].mean(axis=0)

        avg_softmax_pred_dict[filename] = avg_softmax_pred

    print(f"avg_softmax_pred_dict: {avg_softmax_pred_dict}")