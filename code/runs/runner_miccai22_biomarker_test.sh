### Biomarker exp
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# #Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T1a" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2copy_C/checkpoints/epoch=93-step=13911-val_acc=0.8762.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2copy" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2copy_A/checkpoints/last.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2copy" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2copy_B/checkpoints/epoch=73-step=9545-val_acc=0.8598.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2copy" SYSTEM.NUM_GPUS [1] &


#POCOVID
checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2pocovid_C/checkpoints/best.ckpt"
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint DATA.TEST_FOLDS "['E']" DATA.NO_MULTI_CLIP "2" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2pocovid" SYSTEM.NUM_GPUS [1] &
checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2pocovid_A/checkpoints/best.ckpt"
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint DATA.TEST_FOLDS "['E']" DATA.NO_MULTI_CLIP "2" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2pocovid" SYSTEM.NUM_GPUS [1] &
checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2pocovid_B/checkpoints/best.ckpt"
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint DATA.TEST_FOLDS "['E']" DATA.NO_MULTI_CLIP "2" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2pocovid" SYSTEM.NUM_GPUS [1] &



#Abalation Study
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_NoSTAug_T1_C/checkpoints/epoch=31-step=4735-val_acc=0.8687.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_NoSTAug_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_RN34_T1_C/checkpoints/epoch=66-step=9915-val_acc=0.8875.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.BACKBONE "resnet34" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_RN34_T1" SYSTEM.NUM_GPUS [1] &


# #2Fold - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_2Fold_Savg_EquiTS_Crop_T1_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_2Fold_Savg_EquiTS_Crop_T1_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

#RN34
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml MODEL.BACKBONE "resnet34" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_RN34_T1" SYSTEM.NUM_GPUS [1] &

# #I3D - Crop
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_I3D_T1_C/checkpoints/epoch=73-step=10951-val_acc=0.8782.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_I3D_T1_A/checkpoints/epoch=78-step=9637-val_acc=0.8846.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_I3D_T1_B/checkpoints/epoch=93-step=12125-val_acc=0.8681.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [1] &

### End-to-End
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# #Crop
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T1_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2_B/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2copy_C/checkpoints/epoch=55-step=8287-val_acc=0.5903.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2copy" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2copy_A/checkpoints/last.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2copy" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2copy_B/checkpoints/epoch=59-step=7739-val_acc=0.5776.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2copy" SYSTEM.NUM_GPUS [1] &


#POCOVID
checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2pocovid_C/checkpoints/best.ckpt"
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint DATA.TEST_FOLDS "['E']" DATA.NO_MULTI_CLIP "2" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2pocovid" SYSTEM.NUM_GPUS [0] &
checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2pocovid_A/checkpoints/best.ckpt"
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint DATA.TEST_FOLDS "['E']" DATA.NO_MULTI_CLIP "2" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2pocovid" SYSTEM.NUM_GPUS [0] &
checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_T2pocovid_B/checkpoints/best.ckpt"
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint DATA.TEST_FOLDS "['E']" DATA.NO_MULTI_CLIP "2" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2pocovid" SYSTEM.NUM_GPUS [0] &


#Abalation Study
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_NoSTAug_T1_C/checkpoints/epoch=27-step=4143-val_acc=0.5401.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_NoSTAug_T1" SYSTEM.NUM_GPUS [0] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_RN34_T1_C/checkpoints/epoch=21-step=3255-val_acc=0.6056.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.BACKBONE "resnet34" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_RN34_T1" SYSTEM.NUM_GPUS [0] &

# #2Fold - Crop 
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_2Fold_Savg_EquiTS_Crop_T1_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_2Fold_Savg_EquiTS_Crop_T1_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2end_2Fold_Savg_EquiTS_Crop_T1_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# #RN34
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml MODEL.BACKBONE "resnet34"  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_RN34_T1" SYSTEM.NUM_GPUS [0] &

# #I3D - Crop
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_I3D_T1_C/checkpoints/epoch=84-step=12579-val_acc=0.5251.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_I3D_T1_A/checkpoints/epoch=25-step=3171-val_acc=0.6489.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2end_Savg_EquiTS_Crop_I3D_T1_B/checkpoints/epoch=79-step=10319-val_acc=0.5931.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [1] &

### End-to-End SF
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# #Crop
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_T1_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_T2_B/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &


# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_T2a_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T2a" SYSTEM.NUM_GPUS [1] &

# #2Fold - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endSF_2Fold_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endSF_2Fold_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &


# #I3D - Crop
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_I3D_T2_C/checkpoints/epoch=39-step=5399-val_acc=0.3763.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [0] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_I3D_T2_A/checkpoints/epoch=42-step=5675-val_acc=0.5195.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [0] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2endSF_Savg_EquiTS_Crop_I3D_T2_B/checkpoints/epoch=10-step=1638-val_acc=0.3253.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [0] &


### End-to-End Diagnosis
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
#Crop
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T1_B/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_T2_B/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &


# #2Fold - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_2Fold_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# checkpoint="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_End2endDia_2Fold_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# #I3D - Crop
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_I3D_T2_C/checkpoints/epoch=99-step=22399-val_acc=0.2211.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_I3D_T2_A/checkpoints/epoch=27-step=4619-val_acc=0.4975.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [1] &
# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/i3d_LSU-Large-Dataset_End2endDia_Savg_EquiTS_Crop_I3D_T2_B/checkpoints/epoch=53-step=10799-val_acc=0.2424.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "False" MODEL.CHECK_POINT $checkpoint EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [1] &


### Multi-Task
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_multitask.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "MultiTask_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# # #Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_multitask.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "MultiTask_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &


### Seq-Train
# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps_R3/tsm_LSU-Dataset_BioM_AP_EquiTS_PlMl_T1_A/checkpoints/best.ckpt"

# #Crop
# pretrain_freeze="False"

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1a_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T1a" SYSTEM.NUM_GPUS [0] &

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1_A/checkpoints/epoch=41-step=5123-val_acc=0.8851.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1_B/checkpoints/epoch=31-step=4127-val_acc=0.8604.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &


# pretrain_freeze="False"

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &

# checkpoint="/data1/exp_results/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2_B/checkpoints/best.ckpt"
# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_B/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml EXPERIMENT.MODE "Test-generatePreds" MODEL.SAVE_FEATURES "True" MODEL.CHECK_POINT $checkpoint MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &

# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_6MLP_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &

# pretrain_freeze="False"

# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_FREEZE $pretrain_freeze MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_6MLP_AP_EquiTS_PlMl_FineTune_T1" SYSTEM.NUM_GPUS [1] &
