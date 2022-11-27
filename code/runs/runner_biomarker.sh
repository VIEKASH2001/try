
# # Run CMD : 
# python3 code/main.py --cfg code/configs/biomarker_exp1.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RandTS_large_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp1.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RandTS_large_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp1.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RandTS_large_T1" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/biomarker_exp1_baseline.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RandTS_large_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp1_baseline.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RandTS_large_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp1_baseline.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RandTS_large_T1" SYSTEM.NUM_GPUS [0] &


# # Run CMD : Res 224x224px
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RandTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RandTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RandTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &


# # Run CMD : Res 224x224px
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RandTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RandTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RandTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &



# # Run CMD : Res 224x224px - EquiTempSampling
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &


# # Run CMD : Res 224x224px - EquiTempSampling
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &



# # Run CMD : Res 224x224px - EquiTempSampling - TmaxDropSavgCls
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Tmax_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Tmax_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Tmax_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &


# # Run CMD : Res 224x224px - EquiTempSampling - TmaxDropSavgCls
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Baseline_Tmax_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Baseline_Tmax_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Baseline_Tmax_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &



# # Run CMD : Res 224x224px - EquiTempSampling - SavgDropTmaxCls
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &


# # Run CMD : Res 224x224px - EquiTempSampling - SavgDropTmaxCls
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Baseline_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Baseline_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Baseline_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &



# # Run CMD : Res 224x224px - EquiTempSampling - SavgDropTmaxCls
# python3 code/main.py --cfg code/configs/biomarker_exp3.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RN34_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp3.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RN34_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp3.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_RN34_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &


# # Run CMD : Res 224x224px - EquiTempSampling - SavgDropTmaxCls
# python3 code/main.py --cfg code/configs/biomarker_exp3_baseline.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RN34_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp3_baseline.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RN34_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp3_baseline.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_RN34_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &

# Run CMD : Res 224x224px

#Biomarker exp
# python3 code/main.py --cfg code/configs/biomarker_exp4.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "BioM_AP_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp4.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "BioM_AP_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &

#End-to-End
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "End2end_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "End2end_AP_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &

#Multi-Task
# python3 code/main.py --cfg code/configs/biomarker_exp4_multitask.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "MultiTask_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_multitask.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "MultiTask_AP_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [0] &

#Seq-Train
pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps_R3/tsm_LSU-Dataset_BioM_AP_EquiTS_PlMl_T1_A/checkpoints/best.ckpt"
# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps/tsm_LSU-Dataset_BioM_AP_EquiTS_PlMl_T2_A/Biomarker_Exps_Biomarker_Exps/38n9gpuc_0/checkpoints/epoch=21-step=1649.ckpt"
# pretrain_sub="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps_R3/tsm_LSU-Dataset_MultiTask_AP_EquiTS_PlMl_T1_A/checkpoints/best.ckpt"


python3 code/main.py --cfg code/configs/biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "SeqTrain_6MLP_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &

# python3 code/main.py --cfg code/configs/biomarker_exp4_seq_train.yaml SOLVER.EPOCHS "50" MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "SeqTrain_AP_EquiTS_PlMl_Epoch50_T2" SYSTEM.NUM_GPUS [0] &

pretrain_freeze="False"

python3 code/main.py --cfg code/configs/biomarker_exp4_seq_train.yaml MODEL.PRETRAIN_FREEZE $pretrain_freeze MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "SeqTrain_6MLP_AP_EquiTS_PlMl_FineTune_T1" SYSTEM.NUM_GPUS [1] &

# python3 code/main.py --cfg code/configs/biomarker_exp4_seq_train.yaml SOLVER.EPOCHS "50" MODEL.PRETRAIN_FREEZE $pretrain_freeze MODEL.PRETRAIN_SUB $pretrain_sub DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "SeqTrain_AP_EquiTS_PlMl_Epoch50_FineTune_T1" SYSTEM.NUM_GPUS [0] &


# # Variantian testing End-to-End
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Variation_Test_End2end_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Variation_Test_End2end_AP_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Variation_Test_End2end_AP_EquiTS_PlMl_T3" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Variation_Test_End2end_AP_EquiTS_PlMl_T4" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Variation_Test_End2end_AP_EquiTS_PlMl_T5" SYSTEM.NUM_GPUS [1] &

# check_pt="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps/tsm_LSU-Dataset_Variation_Test_End2end_AP_EquiTS_PlMl_T2_A/Biomarker_Exps_Biomarker_Exps/69v0gcr0_0/checkpoints/epoch=7-step=495-val_acc=0.5775.ckpt"
# check_pt="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps/tsm_LSU-Dataset_Variation_Test_End2end_AP_EquiTS_PlMl_T2_A/Biomarker_Exps_Biomarker_Exps/69v0gcr0_0/checkpoints/epoch=16-step=1053-val_acc=0.5623.ckpt"
# check_pt="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps/tsm_LSU-Dataset_Variation_Test_End2end_AP_EquiTS_PlMl_T2_A/Biomarker_Exps_Biomarker_Exps/69v0gcr0_0/checkpoints/epoch=23-step=1487-val_acc=0.5689.ckpt"
# check_pt="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps/tsm_LSU-Dataset_Variation_Test_End2end_AP_EquiTS_PlMl_T2_A/Biomarker_Exps_Biomarker_Exps/69v0gcr0_0/checkpoints/epoch=26-step=1673-val_acc=0.5591.ckpt"
# check_pt="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps/tsm_LSU-Dataset_Variation_Test_End2end_AP_EquiTS_PlMl_T2_A/Biomarker_Exps_Biomarker_Exps/69v0gcr0_0/checkpoints/epoch=92-step=5765-val_acc=0.5624.ckpt"
# check_pt="/home/grg/Research/DARPA-Pneumothorax/results/Biomarker_Exps/tsm_LSU-Dataset_Variation_Test_End2end_AP_EquiTS_PlMl_T2_A/Biomarker_Exps_Biomarker_Exps/69v0gcr0_0/checkpoints/last.ckpt"

# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml EXPERIMENT.MODE "Test" MODEL.CHECK_POINT $check_pt DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Variation_Test_End2end_AP_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_end2end.yaml EXPERIMENT.MODE "TestLast" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Variation_Test_End2end_AP_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [0] &

# python3 code/main.py --cfg code/configs/biomarker_exp4_multitask.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "MultiTask_AP_EquiTS_PlMl_T3" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_multitask.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/biomarker_exp4_multitask.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [1] &


# # Run CMD : Res 224x224px - EquiTempSampling
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/biomarker_exp2_baseline.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Baseline_AP_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
