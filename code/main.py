"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-01 22:42:30
 * @modify date 2022-07-18 21:55:08
 * @desc [description]
 """
 
import numpy as np
import torch

import os
import argparse
import sys

import utils

import wandb

 
from configs.config import get_cfg_defaults

import train
import train_frame

from model.video_models import video_models, i3d, tsm, tsm_seg, tsm_unet_seg
from model import unet

### CUDA Debug Flags ###
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def parse_args():
    """
    [Code refered from FAIR's SlowFast codebase - https://github.com/facebookresearch/SlowFast/blob/9839d1318c0ae17bd82c6a121e5640aebc67f126/slowfast/utils/parser.py#L13]
    Parse the following arguments for a default parser for LungUS AI users.
    Args:
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide LungUS AI video training and testing pipeline."
    )
    
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="code/configs/exp.yaml",
        type=str,
    )
    
    parser.add_argument(
        "opts",
        help="See code/config/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def loadConfig():
    
    args = parse_args()

    # Load default config
    cfg = get_cfg_defaults()
    
    # Merge Experiment config
    # cfg.merge_from_file("experiment.yaml")
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    
    # Merge cmd args
    # opts = ["SYSTEM.NUM_GPUS", 8, "TRAIN.SCALES", "(1, 2, 3, 4)"]
    # cfg.merge_from_list(opts)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Add constuctive configs
    cfg.EXPERIMENT.ROOT_DIR = os.path.join("results", cfg.EXPERIMENT.NAME)
    cfg.EXPERIMENT.RUN_NAME = f"{cfg.EXPERIMENT.MODEL}_{cfg.EXPERIMENT.DATASET}_{cfg.EXPERIMENT.TRIAL}"
    cfg.EXPERIMENT.FOLD_NAME = f"{cfg.EXPERIMENT.RUN_NAME}_{cfg.DATA.VAL_FOLDS[0]}"
    cfg.EXPERIMENT.DIR = os.path.join(cfg.EXPERIMENT.ROOT_DIR, cfg.EXPERIMENT.FOLD_NAME)
    # cfg.EXPERIMENT.ANALYSIS_DIR = os.path.join(cfg.EXPERIMENT.DIR, "videoGradCAM")
    cfg.EXPERIMENT.GRAD_CAM_DIR = os.path.join(cfg.EXPERIMENT.DIR, "videoGradCAM")
    cfg.EXPERIMENT.CHECKPOINT_DIR = os.path.join(cfg.EXPERIMENT.DIR, "checkpoints")

    # Freeze the config
    cfg.freeze()

    print(cfg)

    return cfg


def createExpDirs(cfg):

    if cfg.EXPERIMENT.MODE == "Train":

        if cfg.EXPERIMENT.DEBUG:
            utils.removeDir(cfg.EXPERIMENT.DIR)

        utils.createDir(cfg.EXPERIMENT.DIR)
    
    # #Create tensorboard dir
    # utils.createDir(constants.tensorboard_dir_path, exist_ok=True)


def initModel(cfg):

    if cfg.EXPERIMENT.MODEL == "tsm" and cfg.MODEL.MULTI_TASK:
        model = tsm.TSN(
                    num_class = cfg.DATA.NUM_CLASS + cfg.DATA.NUM2_CLASS, 
                    num_channels = 1 + cfg.DATA.NUM_MASKS,
                    num_segments = cfg.VIDEO_DATA.CLIP_WIDTH, #16, 18, #36, #18,
                    modality = cfg.MODEL.MODALITY,
                    base_model = cfg.MODEL.BACKBONE,
                    pretrain = cfg.MODEL.PRETRAIN,
                    is_shift = True, 
                    shift_div = 8, 
                    shift_place = 'blockres', #'block',
                    partial_bn = False,
                    dropout = cfg.MODEL.DROPOUT,
                    st_consensus_type = cfg.MODEL.ST_CONCENSUS
                )
    elif cfg.EXPERIMENT.MODEL == "tsm" and cfg.MODEL.SEQ_TRAIN:
        model = tsm.TSN(
                    num_class = cfg.MODEL.NUM_SEQ_CLASS, 
                    num_channels = 1 + cfg.DATA.NUM_MASKS,
                    num_segments = cfg.VIDEO_DATA.CLIP_WIDTH, #16, 18, #36, #18,
                    modality = cfg.MODEL.MODALITY,
                    base_model = cfg.MODEL.BACKBONE,
                    pretrain = cfg.MODEL.PRETRAIN,
                    is_shift = True, 
                    shift_div = 8, 
                    shift_place = 'blockres', #'block',
                    partial_bn = False,
                    dropout = cfg.MODEL.DROPOUT,
                    st_consensus_type = cfg.MODEL.ST_CONCENSUS,
                    seq_train = cfg.MODEL.SEQ_TRAIN,
                    num_seq_class = cfg.DATA.NUM_CLASS,
                )

        #Load model weights
        if cfg.MODEL.PRETRAIN_SUB is not None:

            print(f"Loading model weights from {cfg.MODEL.PRETRAIN_SUB}")
                
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_SUB)
            
            model_state_dict = checkpoint['state_dict']
            
            new_model_state_dict = {}
            for k, v in model_state_dict.items():
                new_model_state_dict[".".join(k.split(".")[1:])] = v
            model_state_dict = new_model_state_dict

            if model_state_dict["consensus.fc.weight"].shape[0] > cfg.MODEL.NUM_SEQ_CLASS:
                print(f"Loaded multi-task model weights, so clipping the classifier weigths to {cfg.MODEL.NUM_SEQ_CLASS} by [4:, :]")
                model_state_dict["consensus.fc.weight"] = model_state_dict["consensus.fc.weight"][4:,:]
                model_state_dict["consensus.fc.bias"] = model_state_dict["consensus.fc.bias"][4:]

            model_load_msg = model.load_state_dict(model_state_dict, strict = False)
            print(f"Loaded model weights : {model_load_msg}")

            if cfg.MODEL.PRETRAIN_FREEZE:
                for name, param in model.named_parameters():
                    #Don't freeze the sequential classifier layer
                    if "seq_fc" in name:
                        continue

                    param.requires_grad = False
                
                print(f"Froze weights of model except model.seq_fc")
            
    elif cfg.EXPERIMENT.MODEL == "tsm":
        model = tsm.TSN(
                    num_class = cfg.DATA.NUM_CLASS, 
                    num_channels = 1 + cfg.DATA.NUM_MASKS,
                    num_segments = cfg.VIDEO_DATA.CLIP_WIDTH, #16, 18, #36, #18,
                    modality = cfg.MODEL.MODALITY,
                    base_model = cfg.MODEL.BACKBONE,
                    pretrain = cfg.MODEL.PRETRAIN,
                    is_shift = True, 
                    shift_div = 8, 
                    shift_place = 'blockres', #'block',
                    partial_bn = False,
                    dropout = cfg.MODEL.DROPOUT,
                    st_consensus_type = cfg.MODEL.ST_CONCENSUS
                )        
    elif cfg.EXPERIMENT.MODEL == "i3d":
        model = i3d.pytorchI3D(
                    input_channel = 1 + cfg.DATA.NUM_MASKS,
                    num_classes = cfg.DATA.NUM_CLASS, 
                    pretrained = cfg.MODEL.PRETRAIN == "imagenet",
                )
        
        # model = i3d.I3D(
        #             input_channel = 1 + cfg.DATA.NUM_MASKS,
        #             num_classes = cfg.DATA.NUM_CLASS, 
        #             dropout_keep_prob = 1, 
        #             spatial_squeeze = True,
        #         )

    elif cfg.EXPERIMENT.MODEL == "tsm_seg":
        model = tsm_seg.TSN(
                    num_class = cfg.DATA.NUM_CLASS, 
                    num_seg_class = cfg.DATA.SEG_NUM_CLASS, 
                    num_channels = 1 + cfg.DATA.NUM_MASKS,
                    num_segments = cfg.VIDEO_DATA.CLIP_WIDTH, #16, 18, #36, #18,
                    modality = cfg.MODEL.MODALITY,
                    base_model = cfg.MODEL.BACKBONE,
                    pretrain = cfg.MODEL.PRETRAIN,
                    is_shift = True, 
                    shift_div = 8, 
                    shift_place = 'blockres', #'block',
                    partial_bn = False,
                    dropout = cfg.MODEL.DROPOUT,
                    st_consensus_type = cfg.MODEL.ST_CONCENSUS
                )
    elif cfg.EXPERIMENT.MODEL == "tsm_unet_seg":
        model = tsm_unet_seg.TSN(
                    num_class = cfg.DATA.NUM_CLASS, 
                    num_seg_class = cfg.DATA.SEG_NUM_CLASS, 
                    num_channels = 1 + cfg.DATA.NUM_MASKS,
                    num_segments = cfg.VIDEO_DATA.CLIP_WIDTH, #16, 18, #36, #18,
                    modality = cfg.MODEL.MODALITY,
                    base_model = cfg.MODEL.BACKBONE,
                    pretrain = cfg.MODEL.PRETRAIN,
                    is_shift = True, 
                    shift_div = 8, 
                    shift_place = 'blockres', #'block',
                    partial_bn = False,
                    dropout = cfg.MODEL.DROPOUT,
                    st_consensus_type = cfg.MODEL.ST_CONCENSUS
                )

        #Load model weights
        if cfg.MODEL.PRETRAIN_FREEZE is not None:

            print(f"Loading model weights from {cfg.MODEL.PRETRAIN_FREEZE}")
                
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_FREEZE)
            
            model_state_dict = checkpoint['state_dict']
            
            new_model_state_dict = {}
            for k, v in model_state_dict.items():
                new_model_state_dict[k.replace("model.", "")] = v
            model_state_dict = new_model_state_dict

            model_load_msg = model.base_unet.load_state_dict(model_state_dict)
            print(f"Loaded model weights : {model_load_msg}")

            for p in model.base_unet.parameters():
                p.requires_grad = False
            
            print(f"Froze weights of model.base_unet")

        #Load model weights
        if cfg.MODEL.PRETRAIN_SUB is not None:

            print(f"Loading model weights from {cfg.MODEL.PRETRAIN_SUB}")
                
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_SUB)
            
            model_state_dict = checkpoint['state_dict']
            
            new_model_state_dict = {}
            for k, v in model_state_dict.items():
                new_model_state_dict[k.replace("model.", "")] = v
            model_state_dict = new_model_state_dict

            model_load_msg = model.base_unet.load_state_dict(model_state_dict)
            print(f"Loaded model weights : {model_load_msg}")

    elif cfg.EXPERIMENT.MODEL == "unet":
        model = unet.unet(
                    feature_scale = 1, 
                    in_channels = 1 + cfg.DATA.NUM_MASKS,
                    # n_classes = cfg.DATA.NUM_CLASS,
                    n_classes = cfg.DATA.SEG_NUM_CLASS,
                )

        #Load model weights
        if cfg.MODEL.PRETRAIN is not None:

            print(f"Loading model weights from {cfg.MODEL.PRETRAIN}")
                
            checkpoint = torch.load(cfg.MODEL.PRETRAIN)
            
            model_state_dict = checkpoint['model']
            del model_state_dict["final.weight"]
            del model_state_dict["final.bias"]
            
            model_load_msg = model.load_state_dict(model_state_dict, strict = False)
            print(f"Loaded model weights : {model_load_msg}")

    elif cfg.EXPERIMENT.MODEL == "unetSm":
        model = unet.unetSmall(
                    feature_scale = 1, 
                    in_channels = 1 + cfg.DATA.NUM_MASKS,
                    # n_classes = cfg.DATA.NUM_CLASS,
                    n_classes = cfg.DATA.SEG_NUM_CLASS,
                )

        #Load model weights
        if cfg.MODEL.PRETRAIN is not None:

            print(f"Loading model weights from {cfg.MODEL.PRETRAIN}")
                
            checkpoint = torch.load(cfg.MODEL.PRETRAIN)
            
            model_state_dict = checkpoint['model']
            del model_state_dict["final.weight"]
            del model_state_dict["final.bias"]
            
            model_load_msg = model.load_state_dict(model_state_dict, strict = False)
            print(f"Loaded model weights : {model_load_msg}")

    else:
        raise ValueError(f"Unsupported cfg.EXPERIMENT.MODEL = {cfg.EXPERIMENT.MODEL}!")

    return model


def main():
    
    #Load Config
    cfg = loadConfig()

    #Create exp dir
    createExpDirs(cfg)

    #Write the config
    with open(os.path.join(cfg.EXPERIMENT.DIR, "config.yaml"), "w") as file1:
        file1.write(str(cfg))

    model = initModel(cfg)

    if cfg.MODEL.TYPE == "video":    
        train.cli_main(cfg, model)
    elif cfg.MODEL.TYPE == "frame":    
        train_frame.cli_main(cfg, model)
    else:
        raise ValueError(f"Unsupported cfg.MODEL.TYPE = {cfg.MODEL.TYPE}!")

# Run CMD : 
# python3 code/main.py --cfg code/configs/exp2.yaml DATA.TRAIN_FOLDS "['C', 'B']" DATA.VAL_FOLDS "['A']"
# python3 code/main.py --cfg code/configs/exp2.yaml DATA.TRAIN_FOLDS "['A', 'C']" DATA.VAL_FOLDS "['B']"
# python3 code/main.py --cfg code/configs/exp2.yaml DATA.TRAIN_FOLDS "['A', 'B']" DATA.VAL_FOLDS "['C']"

if __name__ == "__main__":

    print("Started...")
    main()
    print("Finished!")