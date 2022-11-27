### Biomarker exp
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# #Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# #2Fold - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
#RN34
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml MODEL.BACKBONE "resnet34" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_RN34_T1" SYSTEM.NUM_GPUS [1] &
#NoSTAug - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_NoSTAug_T1" SYSTEM.NUM_GPUS [0] &
# #RN34 - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4.yaml VIDEO_DATA.TYPE "crop" MODEL.BACKBONE "resnet34" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_RN34_T1" SYSTEM.NUM_GPUS [2] &

### End-to-End
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# #Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# #2Fold - Crop 
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_2Fold_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# #RN34
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml MODEL.BACKBONE "resnet34"  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_RN34_T1" SYSTEM.NUM_GPUS [0] &
#NoSTAug - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_NoSTAug_T1" SYSTEM.NUM_GPUS [1] &
# #RN34 - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2end.yaml VIDEO_DATA.TYPE "crop" MODEL.BACKBONE "resnet34"  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_RN34_T1" SYSTEM.NUM_GPUS [3] &

### End-to-End SF
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
#Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T2a" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# #2Fold - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
#I3D - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [2] &

python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [2] &
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endSF.yaml EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endSF_Savg_EquiTS_Crop_I3D_T2" SYSTEM.NUM_GPUS [3] &

### End-to-End Diagnosis
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
#Crop
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
# #2Fold - Crop
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_2Fold_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
# #I3D - Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_end2endDia.yaml EXPERIMENT.MODEL "i3d" VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2endDia_Savg_EquiTS_Crop_I3D_T1" SYSTEM.NUM_GPUS [3] &

# ### Multi-Task
# # python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_multitask.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "MultiTask_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# # #Crop
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_multitask.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "MultiTask_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_multitask.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "MultiTask_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_multitask.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "MultiTask_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [1] &


### Seq-Train


# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_6MLP_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# #Crop
# pretrain_freeze="False"

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1a_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T1a" SYSTEM.NUM_GPUS [0] &

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1_A/checkpoints/epoch=41-step=5123-val_acc=0.8851.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T1_B/checkpoints/epoch=31-step=4127-val_acc=0.8604.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T1" SYSTEM.NUM_GPUS [0] &



# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_A/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/MICCAI22_Biomarker_Exps/tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_B/checkpoints/best.ckpt"
# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub MODEL.PRETRAIN_FREEZE $pretrain_freeze VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_FineTune_3MLP_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &


# python3 code/main.py --cfg code/configs/miccai22_biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_FREEZE $pretrain_freeze MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SeqTrain_6MLP_AP_EquiTS_PlMl_FineTune_T1" SYSTEM.NUM_GPUS [1] &
