"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-01 22:41:44
 * @modify date 2022-02-24 19:59:24
 * @desc [description]
 """
  

# Code referenced from YACS codebase: https://github.com/rbgirshick/yacs

from yacs.config import CfgNode as CN


_C = CN()



# Experiment configs
_C.EXPERIMENT = CN()
# Name of the experiment
_C.EXPERIMENT.NAME = "Test"
# Model used for exp
_C.EXPERIMENT.MODEL = "tsm"
# Dataset used for exp
_C.EXPERIMENT.DATASET = "LSU-Dataset"
# Mode of the exp
_C.EXPERIMENT.MODE = "Train"
# Trial no of the exp
_C.EXPERIMENT.TRIAL = "T1"
# Debug mode set while coding
_C.EXPERIMENT.DEBUG = False



# System configs
_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
# _C.SYSTEM.NUM_GPUS = 2
_C.SYSTEM.NUM_GPUS = [0,1]
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 16
# Dataloader pin memory flag
_C.SYSTEM.PIN_MEMORY = True



# Model configs
_C.MODEL = CN()
_C.MODEL.PRETRAIN = None
# Load sub-model module weights 
_C.MODEL.PRETRAIN_SUB = None
# Load and freeze the model weights 
_C.MODEL.PRETRAIN_FREEZE = False

_C.MODEL.BACKBONE = "resnet18"
_C.MODEL.MODALITY = "RGB"
#Model type - video or frame - ["video", "frame"] 
_C.MODEL.TYPE = "video"
_C.MODEL.CHECK_POINT = ""
# #Set True if to take concensus i.e. temporal pooling at the feature level (encoder output) instead of at the logits level (FC-classifier output)  
# _C.MODEL.FEATURE_CONCENSUS = True
# Space-Time concensus/aggregator module - ["AttentionPooling", "VideoTransformer", "TmaxDropSavgCls", "TmaxSavgDropCls", "SavgDropTmaxCls", "SavgDropClsTavg"]
_C.MODEL.ST_CONCENSUS = "TmaxDropSavgCls"

# Dropout layer prob
_C.MODEL.DROPOUT = 0.5 #0.8 #TSM Network dropout at the end

#Multi-task Training 
_C.MODEL.MULTI_TASK = False

#Sequential Training
_C.MODEL.SEQ_TRAIN = False
_C.MODEL.NUM_SEQ_CLASS = -1

#Save resnet features
_C.MODEL.SAVE_FEATURES = True

# Dataset configs
_C.DATA = CN()
# Dataset root dir
_C.DATA.ROOT_PATH = "/data1/datasets/"
# Reprocess the preprocess steps
_C.DATA.REPROCESS = False
# Augment data
_C.DATA.AUGMENT = True
# UpSample data for class balance
_C.DATA.UPSAMPLE = True
# UpSample val set data for class balance
_C.DATA.UPSAMPLE_VAL = True
# Multi-clip evaluation for testing and validation sets
_C.DATA.MULTI_CLIP_EVAL = True
# No of clip-segments to evaluate per video
_C.DATA.NO_MULTI_CLIP = 5
#  Resize image height and width
_C.DATA.IMAGE_RESIZE = (224, 224)

#Input masks
_C.DATA.NUM_MASKS = 0
_C.DATA.MASK1 = ""
_C.DATA.MASK2 = ""

#Segmentation Labels
_C.DATA.SEG_LABEL = ""
# Segmentation No of classes
_C.DATA.SEG_NUM_CLASS = 6
#  Resize seg label image height and width
_C.DATA.SEG_IMAGE_RESIZE = ()
# Segmentation Label names for reports
_C.DATA.SEG_LABEL_NAMES = [ 'bck-grd', 'pl-pneumo', 'pl-healthy', 'vessel', 'chest-wall', 'rib', ]
# Segmentation Label color mapping for saving
_C.DATA.SEG_LABEL_COLORS_DICT = [
                                    (0, 0, 0), #0 #000000 - Background
                                    (242, 5, 246), #1 #F205F6 - pleural line pneumothorax
                                    (255, 0, 0), #2 #FF0000 - pleural line normal
                                    (0, 255, 0), #3 #00FF00 - vessel
                                    (42, 125, 209), #4 #2A7DD1 - chest wall muscle
                                    # 4: (42, 135, 209), #4 #2A7DD1
                                    (221, 255, 51), #5 #DDFF33 - rib bone
                                    # 5: (209, 135, 42), #5
                                ]
# _C.DATA.SEG_LABEL_COLORS_DICT = [
#                                     (0, 0, 0), #0 #000000 - Background
#                                     (246, 5, 242), #1 #F205F6 - pleural line pneumothorax
#                                     (0, 0, 255), #2 #FF0000 - pleural line normal
#                                     (0, 255, 0), #3 #00FF00 - vessel
#                                     (209, 125, 42), #4 #2A7DD1 - chest wall muscle
#                                     # 4: (42, 135, 209), #4 #2A7DD1
#                                     (51, 255, 221), #5 #DDFF33 - rib bone
#                                     # 5: (209, 135, 42), #5
#                                 ]
# _C.DATA.SEG_LABEL_COLORS_DICT = {
#                                     0: (0, 0, 0), #000000 - Background
#                                     1: (246, 5, 242), #F205F6 - pleural line pneumothorax
#                                     2: (0, 0, 255), #FF0000 - pleural line normal
#                                     3: (0, 255, 0), #00FF00 - vessel
#                                     4: (209, 125, 42), #2A7DD1 - chest wall muscle
#                                     # 4: (42, 135, 209), #2A7DD1
#                                     5: (51, 255, 221), #DDFF33 - rib bone
#                                     # 5: (209, 135, 42),
#                                 }
                
# Datasplits
_C.DATA.TEST_SPLIT = 0.2
_C.DATA.NUM_FOLDS = 3
# _C.DATA.TRAIN_FOLDS = ["A", "B"]
# _C.DATA.VAL_FOLDS = ["C"]
_C.DATA.TRAIN_FOLDS = ["C", "B"]
_C.DATA.VAL_FOLDS = ["A"]
_C.DATA.TEST_FOLDS = ["D"]
# _C.DATA.TRAIN_FOLDS = ["A", "C"]
# _C.DATA.VAL_FOLDS = ["B"]
_C.DATA.RANDOM_SPLIT_TRIAL = "R1"

# Label configs



# Dataset labels dict file
_C.DATA.LABEL_DICT = "user_label_gautamgare_12.json"
# No of classes
_C.DATA.NUM_CLASS = 4
# No of classes
_C.DATA.NUM2_CLASS = 4
# Label names for reports
_C.DATA.LABEL_NAMES = [ 'score-0', 'score-1', 'score-2', 'score-3', ]
# Label features to use
_C.DATA.LABEL_FEATURES = ['lung-severity', ]
# Label names for reports
_C.DATA.LABEL2_NAMES = [ ]
# Label features to use
_C.DATA.LABEL2_FEATURES = [ ]
# Probe type to use - ["Linear", "C19-lung", "C19-cardiac"]
_C.DATA.PROBE = "Linear" 



# Video configs
_C.VIDEO_DATA = CN()
# Video type - ["crop_image", "linear_probe_straighten_pleural_line",
#               "linear_probe_under_pleural_line", "plerual_line", "subQ",
#               "subQ_with_pleural_line", "merlin"]
# Video type - ["crop", "mask_subq"]
_C.VIDEO_DATA.TYPE = "crop"

# Video Clip width
_C.VIDEO_DATA.CLIP_WIDTH = 16
# Temporal sampling method - ["equi_temp_sampling", "rand_temp_sampling"]
_C.VIDEO_DATA.TEMP_SAMPLING_METHOD = "equi_temp_sampling"



# Test Video configs
_C.VIDEO_TEST_DATA = CN()
_C.VIDEO_TEST_DATA.LABEL_DICT = []

# Optimizer params
_C.SOLVER = CN()
# A very important hyperparameter
_C.SOLVER.LOSS_FUNC = "cross_entropy"
#  Optimizer to use - Adam, SGD [adam, sgd]
_C.SOLVER.OPTIMIZER = "adam"
#  SCHEDULER to use - ReduceLROnPlateau, CosineAnnealingWarmRestarts [plateau, cosine]
_C.SOLVER.SCHEDULER = "plateau"
# Initial learning rate
_C.SOLVER.INIT_LR = 0.001
# No of epochs to train
_C.SOLVER.EPOCHS = 100
# Train batch size
_C.SOLVER.TRAIN_BATCH_SIZE = 8
# Test & Val batch size
_C.SOLVER.TEST_BATCH_SIZE = 8*2


# Post model training analysis params
_C.POST_ANALYSIS = CN()
# Fit ML classifier models on top
_C.POST_ANALYSIS.FIT_ML_MODELS = True

# videoGradCAM params
_C.CAM = CN()
# Grad CAm enable flag
_C.CAM.ENABLE = True
# A very important hyperparameter
_C.CAM.TRAIN_FREQ = 40 #-1 #TODO - GRG: Don't enable gradCAM for training set as it sets model.eval() mode!
_C.CAM.VAL_FREQ = 40 #10 #5
_C.CAM.TEST_FREQ = 1


#Logger
_C.LOGGER = CN()
# Whether to enable Weights&Biases logging ["online", "offline", "disabled"]
_C.LOGGER.WANDB = "disabled"



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
