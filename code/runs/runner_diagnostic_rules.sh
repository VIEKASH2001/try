### Biomarker exp
# python3 code/main.py --cfg code/configs/diagnosticRules_biomarker.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/diagnosticRules_biomarker.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/diagnosticRules_biomarker.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_PlMl_T2" SYSTEM.NUM_GPUS [1] &
# #Crop
# python3 code/main.py --cfg code/configs/diagnosticRules_biomarker.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Bio_Crop_upsampleVal_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/diagnosticRules_biomarker.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Bio_Crop_upsampleVal_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/diagnosticRules_biomarker.yaml VIDEO_DATA.TYPE "crop" DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Bio_Crop_upsampleVal_T1" SYSTEM.NUM_GPUS [0] &


### End-to-End
# python3 code/main.py --cfg code/configs/diagnosticRules_end2end.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/diagnosticRules_end2end.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/diagnosticRules_end2end.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_PlMl_T1" SYSTEM.NUM_GPUS [0] &
#Crop
python3 code/main.py --cfg code/configs/diagnosticRules_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "E2E_Crop_upsampleVal_T1" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/diagnosticRules_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "E2E_Crop_upsampleVal_T1" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/diagnosticRules_end2end.yaml VIDEO_DATA.TYPE "crop"  DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "E2E_Crop_upsampleVal_T1" SYSTEM.NUM_GPUS [0] &
