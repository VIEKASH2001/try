
# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp1.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "TmaxDropSavgCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp1.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "TmaxDropSavgCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp1.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "TmaxDropSavgCls" &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp1.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SavgDropTmaxCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp1.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SavgDropTmaxCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp1.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SavgDropTmaxCls" &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "PlMl_TmaxDropSavgCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "PlMl_TmaxDropSavgCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "PlMl_TmaxDropSavgCls" &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "PlMl_SavgDropTmaxCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "PlMl_SavgDropTmaxCls" &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "PlMl_SavgDropTmaxCls" &


# # # # Run CMD : 
# # python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Rct_TmaxDropSavgCls_T7" SYSTEM.NUM_GPUS [0] &
# # python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Rct_TmaxDropSavgCls_T7" SYSTEM.NUM_GPUS [0] &
# # python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Rct_TmaxDropSavgCls_T7" SYSTEM.NUM_GPUS [0] &


# # # Run CMD : 
# # python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Rct_SavgDropTmaxCls_T7" SYSTEM.NUM_GPUS [1] &
# # python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Rct_SavgDropTmaxCls_T7" SYSTEM.NUM_GPUS [1] &
# # python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Rct_SavgDropTmaxCls_T7" SYSTEM.NUM_GPUS [1] &

# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "VideoTransformer" EXPERIMENT.TRIAL "VideoTransformer" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "VideoTransformer" EXPERIMENT.TRIAL "VideoTransformer" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "VideoTransformer" EXPERIMENT.TRIAL "VideoTransformer" SYSTEM.NUM_GPUS [1] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "VideoTransformer2" EXPERIMENT.TRIAL "VideoTransformer2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "VideoTransformer2" EXPERIMENT.TRIAL "VideoTransformer2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "VideoTransformer2" EXPERIMENT.TRIAL "VideoTransformer2" SYSTEM.NUM_GPUS [1] &




# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Cosine_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Cosine_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Cosine_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Cosine_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Cosine_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Cosine_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &

# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "VideoTransformer2" EXPERIMENT.TRIAL "Cosine_VideoTransformer2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "VideoTransformer2" EXPERIMENT.TRIAL "Cosine_VideoTransformer2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "VideoTransformer2" EXPERIMENT.TRIAL "Cosine_VideoTransformer2" SYSTEM.NUM_GPUS [1] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling" SYSTEM.NUM_GPUS [1] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Cosine_AttentionPooling" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Cosine_AttentionPooling" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp5.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Cosine_AttentionPooling" SYSTEM.NUM_GPUS [1] &



# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp6.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Mask_Short_SavgDropTmaxCls_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp6.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Mask_Short_SavgDropTmaxCls_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp6.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Mask_Short_SavgDropTmaxCls_T1" SYSTEM.NUM_GPUS [1] &



# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp7.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_RandTS" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp7.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_RandTS" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp7.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_RandTS" SYSTEM.NUM_GPUS [1] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_RandTS_large_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_RandTS_large_T1" SYSTEM.NUM_GPUS [0]
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_RandTS_large_T1" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp9.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_large_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp9.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_large_T1" SYSTEM.NUM_GPUS [1]
# python3 code/main.py --cfg code/configs/pneumothorax_exp9.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "AttentionPooling_large_T1" SYSTEM.NUM_GPUS [1] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SavgDropTmaxCls_RandTS_large_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SavgDropTmaxCls_RandTS_large_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "SavgDropTmaxCls_RandTS_large_T1" SYSTEM.NUM_GPUS [0] &

# Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "TmaxDropSavgCls_RandTS_large_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "TmaxDropSavgCls_RandTS_large_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp8.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "TmaxDropSavgCls_RandTS_large_T1" SYSTEM.NUM_GPUS [1] &

# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp6.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Seg_SavgDropTmaxCls_T3" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp6.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Seg_SavgDropTmaxCls_T3" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp6.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Seg_SavgDropTmaxCls_T3" SYSTEM.NUM_GPUS [1] &



# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp6.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_T4" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp6.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_T4" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp6.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_T4" SYSTEM.NUM_GPUS [1] &



# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp7.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_T5" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp7.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_T5" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp7.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_T5" SYSTEM.NUM_GPUS [1] &



# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp8.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_4cls_T6" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp8.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_4cls_T6" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_seg_exp8.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Seg_AttentionPooling_4cls_T6" SYSTEM.NUM_GPUS [1] &



# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp1.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T2" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp1.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T2" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp1.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T2" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T2" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T2" SYSTEM.NUM_GPUS [1] &




# Run CMD : 
python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp3.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T1" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp3.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T1" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp3.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T1" SYSTEM.NUM_GPUS [0] &


# Run CMD : 
python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp4.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T1" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp4.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T1" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp4.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T1" SYSTEM.NUM_GPUS [1] &



# # # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "PB_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "PB_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "PB_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "PB_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "PB_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "PB_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &

# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Deform_TmaxDropSavgCls_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Deform_TmaxDropSavgCls_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Deform_TmaxDropSavgCls_T1" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Deform_SavgDropTmaxCls_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Deform_SavgDropTmaxCls_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Deform_SavgDropTmaxCls_T1" SYSTEM.NUM_GPUS [1] &


# # # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp4.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Mask_Short_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp4.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Mask_Short_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp4.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Mask_Short_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp4.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Mask_Short_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp4.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Mask_Short_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp4.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Mask_Short_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &


# Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp3.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Flow_Rct_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp3.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Flow_Rct_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp3.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "TmaxDropSavgCls" EXPERIMENT.TRIAL "Flow_Rct_TmaxDropSavgCls" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_exp3.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Flow_Rct_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp3.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Flow_Rct_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_exp3.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "SavgDropTmaxCls" EXPERIMENT.TRIAL "Flow_Rct_SavgDropTmaxCls" SYSTEM.NUM_GPUS [1] &
