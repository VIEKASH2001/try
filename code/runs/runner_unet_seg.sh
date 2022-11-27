
pretrain_freeze_A="/home/grg/Research/DARPA-Pneumothorax/results/Pneumothorax_Exps/unetSm_DARPA-Seg-Dataset_Frame_Seg_4cls_T1_A/Pneumothorax_Exps_Pneumothorax_Exps/15xoddmy_0/checkpoints/epoch=1-step=969.ckpt"
pretrain_freeze_B="/home/grg/Research/DARPA-Pneumothorax/results/Pneumothorax_Exps/unetSm_DARPA-Seg-Dataset_Frame_Seg_4cls_T1_B/Pneumothorax_Exps_Pneumothorax_Exps/17qvfkdl_0/checkpoints/epoch=3-step=1823.ckpt"
pretrain_freeze_C="/home/grg/Research/DARPA-Pneumothorax/results/Pneumothorax_Exps/unetSm_DARPA-Seg-Dataset_Frame_Seg_4cls_T1_C/Pneumothorax_Exps_Pneumothorax_Exps/2gfgtl6q_0/checkpoints/epoch=12-step=5940.ckpt"

# Run CMD : 
python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp1.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.PRETRAIN_FREEZE $pretrain_freeze_A EXPERIMENT.TRIAL "TSMunet_Seg_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp1.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.PRETRAIN_FREEZE $pretrain_freeze_B EXPERIMENT.TRIAL "TSMunet_Seg_T2" SYSTEM.NUM_GPUS [0] &
python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp1.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.PRETRAIN_FREEZE $pretrain_freeze_C EXPERIMENT.TRIAL "TSMunet_Seg_T2" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" EXPERIMENT.TRIAL "TSMunet_EnE_Seg_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" EXPERIMENT.TRIAL "TSMunet_EnE_Seg_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" EXPERIMENT.TRIAL "TSMunet_EnE_Seg_T1" SYSTEM.NUM_GPUS [0] &


# Run CMD : 
python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.PRETRAIN_SUB $pretrain_freeze_A EXPERIMENT.TRIAL "TSMunet_EnE_Seg_PreTrain_T1" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.PRETRAIN_SUB $pretrain_freeze_B EXPERIMENT.TRIAL "TSMunet_EnE_Seg_PreTrain_T1" SYSTEM.NUM_GPUS [1] &
python3 code/main.py --cfg code/configs/pneumothorax_unet_seg_exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.PRETRAIN_SUB $pretrain_freeze_C EXPERIMENT.TRIAL "TSMunet_EnE_Seg_PreTrain_T1" SYSTEM.NUM_GPUS [1] &



# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp3.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp3.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T1" SYSTEM.NUM_GPUS [0] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp3.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_T1" SYSTEM.NUM_GPUS [0] &


# # Run CMD : 
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp4.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp4.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T1" SYSTEM.NUM_GPUS [1] &
# python3 code/main.py --cfg code/configs/pneumothorax_frame_seg_exp4.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']" MODEL.ST_CONCENSUS "AttentionPooling" EXPERIMENT.TRIAL "Frame_Seg_4cls_ISBI_pretrain_T1" SYSTEM.NUM_GPUS [1] &


