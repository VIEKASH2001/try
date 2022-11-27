### Biomarker exp

# #Crop
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_bmi.yaml DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_BMI_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_bmi.yaml DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_BMI_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_bmi.yaml DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "BioM_Savg_EquiTS_Crop_BMI_T2" SYSTEM.NUM_GPUS [0] &

### End-to-End

# #Crop
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end_bmi.yaml DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_BMI_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end_bmi.yaml DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_BMI_T2" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/tmi_biomarker_exp1_end2end_bmi.yaml DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "End2end_Savg_EquiTS_Crop_BMI_T2" SYSTEM.NUM_GPUS [1] &
