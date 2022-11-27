### Biomarker exp

# #Crop
# python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [0] &

# #Crop less P30
# python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.RANDOM_SPLIT_TRIAL "R1P30" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_P30_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.RANDOM_SPLIT_TRIAL "R1P30" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_P30_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.RANDOM_SPLIT_TRIAL "R1P30" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_P30_T2" SYSTEM.NUM_GPUS [0] &

# #Crop less P60
# python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.RANDOM_SPLIT_TRIAL "R1P60" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_P60_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.RANDOM_SPLIT_TRIAL "R1P60" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_P60_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1.yaml DATA.RANDOM_SPLIT_TRIAL "R1P60" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_P60_T2" SYSTEM.NUM_GPUS [0] &

### End-to-End

# #Crop
# python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_T2" SYSTEM.NUM_GPUS [1] &

# #Crop less P30
# python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml  DATA.RANDOM_SPLIT_TRIAL "R1P30" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_P30_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml DATA.RANDOM_SPLIT_TRIAL "R1P30" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_P30_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml DATA.RANDOM_SPLIT_TRIAL "R1P30" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_P30_T2" SYSTEM.NUM_GPUS [1] &

# #Crop less P60
# python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml  DATA.RANDOM_SPLIT_TRIAL "R1P60" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_P60_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml DATA.RANDOM_SPLIT_TRIAL "R1P60" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_P60_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end.yaml DATA.RANDOM_SPLIT_TRIAL "R1P60" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_P60_T2" SYSTEM.NUM_GPUS [1] &
