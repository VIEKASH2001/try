"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-01 22:43:12
 * @modify date 2022-02-01 15:20:29
 * @desc [description]
 """
 

import torch
from torch._C import device 

from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers

from analysis.videoGradCAM import VideoGradCAM
from analysis.visualizeAnnotation import saveGradCAMPredAsGIF, savePredAsGIF, saveSegPredAsGIF
from analysis.analysisPlots import plot_confusion_matrix
from analysis.segmentation_metrics import MetricEvaluation

from data import video_dataset
from data import video_seg_dataset




def computeLoss(cfg, outputs, targets):

    if cfg.SOLVER.LOSS_FUNC == "cross_entropy":
        loss = F.cross_entropy(outputs["pred_cls"], targets["lb_cls"])

    elif cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
        loss = F.binary_cross_entropy_with_logits(outputs["pred_cls"], targets["lb_cls"])

    elif cfg.SOLVER.LOSS_FUNC ==  "cross_entropy+binary_cross_entropy":
        alpha = 0.5
        beta = 1.0
        # loss = alpha * F.cross_entropy(outputs["pred_cls"], targets["lb_cls"]) + (1 - alpha) * F.cross_entropy(outputs["pred_seg"], targets["lb_seg"])
        loss = alpha * F.cross_entropy(outputs["pred_cls"][:, :4], targets["lb_cls"]) \
                + (1 - alpha) * beta * F.binary_cross_entropy_with_logits(outputs["pred_cls"][:, 4:], targets["lb_2_cls"])

        # aa.reshape(aa.shape[0], -1).min(dim = 1)[0]

    elif cfg.SOLVER.LOSS_FUNC ==  "cross_entropy+seg_cross_entropy":
        alpha = 0.5
        beta = 1.0/5
        # loss = alpha * F.cross_entropy(outputs["pred_cls"], targets["lb_cls"]) + (1 - alpha) * F.cross_entropy(outputs["pred_seg"], targets["lb_seg"])
        loss = alpha * F.cross_entropy(outputs["pred_cls"], targets["lb_cls"]) \
                + (1 - alpha) * beta * F.cross_entropy(outputs["pred_seg"].transpose(2,1), targets["lb_seg"].squeeze(2), ignore_index = -1)

        # aa.reshape(aa.shape[0], -1).min(dim = 1)[0]

    else:
        raise ValueError(f"Wrong cfg.SOLVER.LOSS_FUNC = {cfg.SOLVER.LOSS_FUNC}!")

    return loss


import os
import utils

from itertools import repeat
from multiprocessing import Pool

class VideoGradCAMCallback(pl_callbacks.Callback):

    def __init__(self, cfg, path, feature_module, target_layer_names, model_name, class_names, loggers) -> None:
        super().__init__()

        self.cfg = cfg
        self.path = path
        self.class_names = class_names

        self.feature_module = feature_module
        self.target_layer_names = target_layer_names
        self.model_name = model_name
        self.loggers = loggers

        self.trainCAMDir = "trainGramCAM"
        self.valCAMDir = "valGramCAM"
        self.testCAMDir = "testGramCAM"
        
        #Append the root path to dir
        self.trainCAMDir = os.path.join(self.path, self.trainCAMDir)
        self.valCAMDir = os.path.join(self.path, self.valCAMDir)
        self.testCAMDir = os.path.join(self.path, self.testCAMDir)

        self.eval_train_freq = cfg.CAM.TRAIN_FREQ
        self.eval_val_freq = cfg.CAM.VAL_FREQ
        self.eval_test_freq = cfg.CAM.TEST_FREQ

    def _recreatePredDirs(self, dir):

        utils.removeDir(dir)
        # [utils.removeDir(os.path.join(self.path, dir, 'valid', a)) for a in self.class_names]
        # [utils.removeDir(os.path.join(self.path, dir, 'invalid', a)) for a in self.class_names]

        [utils.createDir(os.path.join(dir, 'valid', a), exist_ok=True) for a in self.class_names]
        [utils.createDir(os.path.join(dir, 'invalid', a), exist_ok=True) for a in self.class_names]

    def _visualizeGradCAM(self, model, filenames, clips, targets, preds, epoch, batch_id, path, dataset_type):
       
        videoGradCAM = VideoGradCAM(model, feature_module = self.feature_module, target_layer_names = self.target_layer_names, flow_input = "flow" in self.cfg.VIDEO_DATA.TYPE)

        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        cams, gbs, cam_gbs, grad_preds = videoGradCAM.getCAM(clips)

        assert np.array_equal(preds.argmax(1).cpu().numpy(), grad_preds), "Error! GradCAM prediction not same as training prediction."

        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            target_lbs = [[self.class_names[c] for c in np.where(batch_item == 1)[0]] for batch_item in targets.detach().cpu().numpy()]
            pred_lbs = [[self.class_names[c] for c in np.where(batch_item == 1)[0]] for batch_item in torch.sigmoid(preds.detach().cpu()).numpy()]
        else:
            target_lbs = [self.class_names[c] for c in targets]
            pred_lbs = [self.class_names[c] for c in grad_preds]

        clips = (clips.detach()*255).to(torch.uint8)
        if "flow" in self.cfg.VIDEO_DATA.TYPE:
            clips = clips.permute(0,1,3,4,2)[:,:,:,:,0].unsqueeze(-1).repeat(1,1,1,1,3)
        else:
            # clips = clips.squeeze(2).unsqueeze(-1).repeat(1,1,1,1,3)
            clips = clips.permute(0,1,3,4,2)
            if clips.shape[-1] == 1:
                clips = clips.repeat(1,1,1,1,3)

        clips = clips.detach().cpu().numpy()

        # with Pool() as pool:
        #     results = pool.starmap(videoGradCAM.getCAM, clips)

        # for i, (filename, clip, target, pred) in enumerate(zip(filenames, clips, targets, preds)):

        #     cam, gb, cam_gb, grad_pred = videoGradCAM.getCAM(clip)

            # assert pred.argmax() == grad_pred, "Error! GradCAM prediction not same as training prediction."

        for i, (filename, clip, target_lb, pred_lb, cam, gb, cam_gb) in enumerate(zip(filenames, clips, target_lbs, pred_lbs, cams, gbs, cam_gbs)):

            gif_name = f"{filename}_epoch{epoch}_input{batch_id+i}.gif"

            if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
                gif_path = os.path.join(path, "valid" if target_lb == pred_lb else "invalid", gif_name)
            else:
                gif_path = os.path.join(path, "valid" if target_lb == pred_lb else "invalid", target_lb, gif_name)

            gif_video = saveGradCAMPredAsGIF(clip, cam, gb, cam_gb, target_lb, pred_lb, gif_path, filename = gif_name)
            #Change from TxHxWxC -> BxTxCxxHxW
            gif_video = torch.tensor(gif_video).permute(0,3,1,2).unsqueeze(0)

            if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
                tb_gif_path = os.path.join(dataset_type, "valid" if target_lb == pred_lb else "invalid", f"{filename}_input{batch_id+i}.gif")
            else:
                tb_gif_path = os.path.join(dataset_type, "valid" if target_lb == pred_lb else "invalid", target_lb, f"{filename}_input{batch_id+i}.gif")
            self.loggers[1].experiment.add_video(tb_gif_path, gif_video, epoch)

        torch.set_grad_enabled(prev_grad_state)


    # def _visualizeGradCAMWorker(self, i, filename, clip, target_lb, pred_lb, cam, gb, cam_gb, epoch, batch_id, path, dataset_type):
    # def _visualizeGradCAMWorker(self, i, filename, clip, target_lb, pred_lb, cam, gb, cam_gb, epoch, batch_id, path, dataset_type, logger):
    def _visualizeGradCAMWorker(self, i, filename, clip, target_lb, pred_lb, cam, gb, cam_gb):
  
        
        gif_name = f"{filename}_epoch{self.epoch}_input{self.batch_id+i}.gif"

        gif_path = os.path.join(self.path, "valid" if target_lb == pred_lb else "invalid", target_lb, gif_name)

        gif_video = saveGradCAMPredAsGIF(clip, cam, gb, cam_gb, target_lb, pred_lb, gif_path, filename = gif_name)
        #Change from TxHxWxC -> BxTxCxxHxW
        gif_video = torch.tensor(gif_video).permute(0,3,1,2).unsqueeze(0)

        tb_gif_path = os.path.join(self.dataset_type, "valid" if target_lb == pred_lb else "invalid", target_lb, f"{filename}_input{batch_id+i}.gif")
        self.loggers[1].experiment.add_video(tb_gif_path, gif_video, self.epoch)
        # logger.experiment.add_video(tb_gif_path, gif_video, epoch)


    def _visualizeGradCAM_v1(self, model, filenames, clips, targets, preds, epoch, batch_id, path, dataset_type):
       
        videoGradCAM = VideoGradCAM(model, feature_module = self.feature_module, target_layer_names = self.target_layer_names)

        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        cams, gbs, cam_gbs, grad_preds = videoGradCAM.getCAM(clips)

        assert np.array_equal(preds.argmax(1).cpu().numpy(), grad_preds), "Error! GradCAM prediction not same as training prediction."

        target_lbs = [self.class_names[c] for c in targets]
        pred_lbs = [self.class_names[c] for c in grad_preds]

        clips = (clips.detach()*255).to(torch.uint8)
        clips = clips.squeeze(2).unsqueeze(-1).repeat(1,1,1,1,3)
        clips = clips.detach().cpu().numpy()

        self.epoch = epoch
        self.batch_id = batch_id
        self.path = path
        self.dataset_type = dataset_type
        
        # with Pool() as pool:
        with Pool(5) as pool:
            # results = pool.starmap(self._visualizeGradCAMWorker, 
            #         zip(np.arange(len(filenames)), filenames, clips, target_lbs, pred_lbs, cams, gbs, cam_gbs, 
            #         repeat(epoch), repeat(batch_id), repeat(path), repeat(dataset_type)), repeat(self.loggers[1]))
            results = pool.starmap(self._visualizeGradCAMWorker, 
                    zip(np.arange(len(filenames)), filenames, clips, target_lbs, pred_lbs, cams, gbs, cam_gbs))

        results
           
        torch.set_grad_enabled(prev_grad_state)

    def _visualizeInputPred(self, filenames, clips, targets, preds, epoch, batch_id, path, dataset_type):
       
        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            target_lbs = [[self.class_names[c] for c in np.where(batch_item == 1)[0]] for batch_item in targets.detach().cpu().numpy()]
            pred_lbs = [[self.class_names[c] for c in np.where(batch_item == 1)[0]] for batch_item in torch.sigmoid(preds.detach().cpu()).numpy()]
        else:
            target_lbs = [self.class_names[c] for c in targets]
            pred_lbs = [self.class_names[c] for c in preds.argmax(1).cpu().numpy()]

        clips = (clips.detach()*255).to(torch.uint8)
        
        if "flow" in self.cfg.VIDEO_DATA.TYPE:
            clips = clips.permute(0,1,3,4,2)[:,:,:,:,0].unsqueeze(-1).repeat(1,1,1,1,3)
        else:
            # clips = clips.squeeze(2).unsqueeze(-1).repeat(1,1,1,1,3)
            clips = clips.permute(0,1,3,4,2)
            if clips.shape[-1] == 1:
                clips = clips.repeat(1,1,1,1,3)

        clips = clips.detach().cpu().numpy()

        # with Pool() as pool:
        #     results = pool.starmap(videoGradCAM.getCAM, clips)

        # for i, (filename, clip, target, pred) in enumerate(zip(filenames, clips, targets, preds)):

        #     cam, gb, cam_gb, grad_pred = videoGradCAM.getCAM(clip)

            # assert pred.argmax() == grad_pred, "Error! GradCAM prediction not same as training prediction."

        for i, (filename, clip, target_lb, pred_lb) in enumerate(zip(filenames, clips, target_lbs, pred_lbs)):

            gif_name = f"{filename}_epoch{epoch}_input{batch_id+i}.gif"

            if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
                gif_path = os.path.join(path, "valid" if target_lb == pred_lb else "invalid", gif_name)
            else:
                gif_path = os.path.join(path, "valid" if target_lb == pred_lb else "invalid", target_lb, gif_name)

            gif_video = savePredAsGIF(clip, target_lb, pred_lb, gif_path, filename = gif_name)

            #Change from TxHxWxC -> BxTxCxxHxW
            gif_video = torch.tensor(gif_video).permute(0,3,1,2).unsqueeze(0)

            if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
                tb_gif_path = os.path.join(dataset_type, "valid" if target_lb == pred_lb else "invalid", f"{filename}_input{batch_id+i}.gif")
            else:
                tb_gif_path = os.path.join(dataset_type, "valid" if target_lb == pred_lb else "invalid", target_lb, f"{filename}_input{batch_id+i}.gif")
            
            self.loggers[1].experiment.add_video(tb_gif_path, gif_video, epoch)


    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_start(trainer, pl_module)

        epoch = trainer.current_epoch

        #Save only a few training batch predictions
        # if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0 or epoch == 0:
        if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0:
            return

        # Recreate train-video GramCAM directories
        self._recreatePredDirs(self.trainCAMDir)
    
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_start(trainer, pl_module)

        epoch = trainer.current_epoch

        if self.eval_val_freq == -1 or epoch % self.eval_val_freq > 0 or epoch == 0:
            return
     
        # Recreate val-video GramCAM directories
        self._recreatePredDirs(self.valCAMDir)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_start(trainer, pl_module)

        epoch = trainer.current_epoch

        if self.eval_test_freq == -1 or epoch % self.eval_test_freq > 0 or epoch == 0:
            return

        # Recreate test-video GramCAM directories
        self._recreatePredDirs(self.testCAMDir)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        epoch = trainer.current_epoch
        
        #Save only a few training batch predictions
        # if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0 or batch_idx > 1 or epoch == 0:
        if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0 or batch_idx > 1:
            return

        print("on_train_batch_end")


        filenames, clips, targets_dict = batch
        preds = outputs["pred_cls"]
        targets = targets_dict["lb_cls"]
        # batch_id = batch_idx*
        # loggers = trainer.logger.experiment

        # self._visualizeGradCAM(pl_module.model, filenames, clips, targets, preds, epoch, batch_idx, self.trainCAMDir, "Train")
        self._visualizeInputPred(filenames, clips, targets, preds, epoch, batch_idx, self.trainCAMDir, "Train")

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        epoch = trainer.current_epoch
        
        if self.eval_val_freq == -1 or epoch % self.eval_val_freq > 0 or epoch == 0:
            return
        
        print("on_validation_batch_end")

        filenames, clips, targets_dict = batch
        preds = outputs["pred_cls"]
        targets = targets_dict["lb_cls"]

        # batch_id = batch_idx*
        self._visualizeGradCAM(pl_module.model, filenames, clips, targets, preds, epoch, batch_idx, self.valCAMDir, "Val")

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
        epoch = trainer.current_epoch
        
        if self.eval_test_freq == -1 or epoch % self.eval_test_freq > 0 or epoch == 0:
            return

        print("on_test_batch_end")

        filenames, clips, targets_dict = batch
        preds = outputs["pred_cls"]
        targets = targets_dict["lb_cls"]

        # batch_id = batch_idx*
        self._visualizeGradCAM(pl_module.model, filenames, clips, targets, preds, epoch, batch_idx, self.testCAMDir, "Test")


class VideoSegCallback(pl_callbacks.Callback):

    def __init__(self, cfg, path, class_names, seg_class_names, loggers) -> None:
        super().__init__()

        self.cfg = cfg
        self.path = path
        self.class_names = class_names
        self.seg_class_names = seg_class_names

        self.loggers = loggers

        self.trainSegDir = "trainSeg"
        self.valSegDir = "valSeg"
        self.testSegDir = "testSeg"
        
        #Append the root path to dir
        self.trainSegDir = os.path.join(self.path, self.trainSegDir)
        self.valSegDir = os.path.join(self.path, self.valSegDir)
        self.testSegDir = os.path.join(self.path, self.testSegDir)

        self.eval_train_freq = cfg.CAM.TRAIN_FREQ
        self.eval_val_freq = cfg.CAM.VAL_FREQ
        self.eval_test_freq = cfg.CAM.TEST_FREQ

        self.colormap = np.array(self.cfg.DATA.SEG_LABEL_COLORS_DICT, dtype = np.uint8)

    def _recreatePredDirs(self, dir):

        utils.removeDir(dir)
        # [utils.removeDir(os.path.join(self.path, dir, 'valid', a)) for a in self.class_names]
        # [utils.removeDir(os.path.join(self.path, dir, 'invalid', a)) for a in self.class_names]

        [utils.createDir(os.path.join(dir, 'valid', a), exist_ok=True) for a in self.class_names]
        [utils.createDir(os.path.join(dir, 'invalid', a), exist_ok=True) for a in self.class_names]

    def _visualizeSegPred(self, filenames, clips, seg_targets, seg_preds, targets, preds, epoch, batch_id, path, dataset_type):
       
       
        target_lbs = [self.class_names[c] for c in targets]
        pred_lbs = [self.class_names[c] for c in preds.argmax(1).cpu().numpy()]

        clips = (clips.detach()*255).to(torch.uint8)
        
        #Resize seg targets and preds
        if clips.shape[-2:] != seg_targets.shape[-2:]:
            seg_targets = seg_targets.float()
            seg_targets = F.interpolate(seg_targets, size = [1]+list(clips.shape[-2:]), mode = "nearest")
            seg_targets = seg_targets.long()

            seg_preds = seg_preds.float()
            seg_preds = F.interpolate(seg_preds, size = [1]+list(clips.shape[-2:]), mode = "nearest")
            seg_preds = seg_preds.long()
        
        if "flow" in self.cfg.VIDEO_DATA.TYPE:
            clips = clips.permute(0,1,3,4,2)[:,:,:,:,0].unsqueeze(-1).repeat(1,1,1,1,3)
        else:
            # clips = clips.squeeze(2).unsqueeze(-1).repeat(1,1,1,1,3)
            clips = clips.permute(0,1,3,4,2)
            if clips.shape[-1] == 1:
                clips = clips.repeat(1,1,1,1,3)

            # seg_targets = seg_targets.permute(0,1,3,4,2)
            # if seg_targets.shape[-1] == 1:
            #     seg_targets = seg_targets.repeat(1,1,1,1,3)

            # seg_preds = seg_preds.permute(0,1,3,4,2)
            # if seg_preds.shape[-1] == 1:
            #     seg_preds = seg_preds.repeat(1,1,1,1,3)

            

        clips = clips.detach().cpu().numpy()
        # seg_targets = seg_targets.detach().cpu().numpy()
        # seg_preds = seg_preds.detach().cpu().numpy()
        seg_targets = self.colormap[seg_targets.squeeze(2).detach().cpu().numpy()]
        seg_preds = self.colormap[seg_preds.squeeze(2).detach().cpu().numpy()]

        for i, (filename, clip, seg_target, seg_pred, target_lb, pred_lb) in enumerate(zip(filenames, clips, seg_targets, seg_preds, target_lbs, pred_lbs)):

            gif_name = f"{filename}_epoch{epoch}_input{batch_id+i}.gif"

            gif_path = os.path.join(path, "valid" if target_lb == pred_lb else "invalid", target_lb, gif_name)

            gif_video = saveSegPredAsGIF(clip, seg_target, seg_pred, target_lb, pred_lb, gif_path, filename = gif_name)

            #Change from TxHxWxC -> BxTxCxxHxW
            gif_video = torch.tensor(gif_video).permute(0,3,1,2).unsqueeze(0)

            tb_gif_path = os.path.join(dataset_type + "_Seg", "valid" if target_lb == pred_lb else "invalid", target_lb, f"{filename}_input{batch_id+i}.gif")
            self.loggers[1].experiment.add_video(tb_gif_path, gif_video, epoch)


    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_start(trainer, pl_module)

        epoch = trainer.current_epoch

        #Save only a few training batch predictions
        # if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0 or epoch == 0:
        if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0:
            return

        # Recreate train-video GramCAM directories
        self._recreatePredDirs(self.trainSegDir)
    
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_start(trainer, pl_module)

        epoch = trainer.current_epoch

        if self.eval_val_freq == -1 or epoch % self.eval_val_freq > 0 or epoch == 0:
            return
     
        # Recreate val-video GramCAM directories
        self._recreatePredDirs(self.valSegDir)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_start(trainer, pl_module)

        epoch = trainer.current_epoch

        if self.eval_test_freq == -1 or epoch % self.eval_test_freq > 0 or epoch == 0:
            return

        # Recreate test-video GramCAM directories
        self._recreatePredDirs(self.testSegDir)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        epoch = trainer.current_epoch
        
        #Save only a few training batch predictions
        # if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0 or batch_idx > 1 or epoch == 0:
        if self.eval_train_freq == -1 or epoch % self.eval_train_freq > 0 or batch_idx > 1:
            return

        print("on_train_batch_end")


        filenames, clips, targets_dict = batch
        preds = outputs["pred_cls"]
        targets = targets_dict["lb_cls"]
        # batch_id = batch_idx*
        # loggers = trainer.logger.experiment

        seg_preds = outputs["pred_seg"]
        seg_targets = targets_dict["lb_seg"]

        seg_preds = seg_preds.argmax(dim = 2, keepdim = True)

        self._visualizeSegPred(filenames, clips, seg_targets, seg_preds, targets, preds, epoch, batch_idx, self.trainSegDir, "Train")

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        
        epoch = trainer.current_epoch
        
        if self.eval_val_freq == -1 or epoch % self.eval_val_freq > 0 or epoch == 0:
            return
        
        print("on_validation_batch_end")

        filenames, clips, targets_dict = batch
        preds = outputs["pred_cls"]
        targets = targets_dict["lb_cls"]

        seg_preds = outputs["pred_seg"]
        seg_targets = targets_dict["lb_seg"]

        seg_preds = seg_preds.argmax(dim = 2, keepdim = True)

        # batch_id = batch_idx*
        self._visualizeSegPred(filenames, clips, seg_targets, seg_preds, targets, preds, epoch, batch_idx, self.valSegDir, "Val")

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
        epoch = trainer.current_epoch
        
        if self.eval_test_freq == -1 or epoch % self.eval_test_freq > 0 or epoch == 0:
            return

        print("on_test_batch_end")

        filenames, clips, targets_dict = batch
        preds = outputs["pred_cls"]
        targets = targets_dict["lb_cls"]

        seg_preds = outputs["pred_seg"]
        seg_targets = targets_dict["lb_seg"]

        seg_preds = seg_preds.argmax(dim = 2, keepdim = True)

        # batch_id = batch_idx*
        self._visualizeSegPred(filenames, clips, seg_targets, seg_preds, targets, preds, epoch, batch_idx, self.testSegDir, "Test")


import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix



class ComputeMetricsCallback(pl_callbacks.Callback):

    def __init__(self, cfg, path, labels, class_names, loggers) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.path = path

        self.labels = labels
        self.class_names = class_names
        self.loggers = loggers

        
        self.train_filenames = []
        self.val_filenames = []
        self.test_filenames = []

        self.train_targets = []
        self.val_targets = []
        self.test_targets = []

        self.train_prob_preds = []
        self.val_prob_preds = []
        self.test_prob_preds = []

        self.best_val_acc = 0
        
    def _computeMetrics(self, filenames, targets, prob_preds, dataset_type, epoch):

    

        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            preds = torch.sigmoid(torch.tensor(prob_preds)).numpy() > 0.5
        else:
            preds = np.array(prob_preds).argmax(axis = 1)
        
        accuracy = accuracy_score(targets, preds)

        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            confusionMatrix = multilabel_confusion_matrix(targets, preds, labels = self.labels)
        else:
            confusionMatrix = confusion_matrix(targets, preds, labels = self.labels)
            
        classificationReport = classification_report(targets, preds, labels = self.labels, target_names = self.class_names, digits=5)

        print(f"Epoch-{epoch} {dataset_type} : accuracy = {accuracy}")
        print(f"Epoch-{epoch} {dataset_type} : confusionMatrix = \n {confusionMatrix}")
        print(f"Epoch-{epoch} {dataset_type} : classificationReport = \n {classificationReport}")

        self.loggers[0].experiment.log({f"Accuracy/{dataset_type}": accuracy}) #, epoch)
        self.loggers[1].experiment.add_scalar(f"Accuracy/{dataset_type}", accuracy, epoch)

        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            return accuracy
        
        cm_fig = plot_confusion_matrix(confusionMatrix, target_names = self.class_names, path = self.path, normalize=False, prefix = dataset_type)
        norm_cm_fig = plot_confusion_matrix(confusionMatrix, target_names = self.class_names, path = self.path, normalize=True, prefix = f"{dataset_type}_normalized")

        # self.loggers[0].experiment.log({f"ConfusionMatrix/{dataset_type}": cm_fig}) #, epoch)
        # self.loggers[0].experiment.log({f"ConfusionMatrixNormalized/{dataset_type}": norm_cm_fig}) #, epoch)
        self.loggers[1].experiment.add_figure(f"ConfusionMatrix/{dataset_type}", cm_fig, epoch)
        self.loggers[1].experiment.add_figure(f"ConfusionMatrixNormalized/{dataset_type}", norm_cm_fig, epoch)

        return accuracy

    def _computeConsensusMetrics(self, filenames, targets, prob_preds, dataset_type, epoch):
        
        filenames = np.array(filenames)
        targets = np.array(targets)
        prob_preds = np.array(prob_preds)

        video_filenames = np.unique(filenames)
        
        video_targets = []
        video_prob_preds = []
        for video in video_filenames:

            video_idx = np.where(np.array(filenames) == video)[0]
            
            target = targets[video_idx][0]
            prob_pred = prob_preds[video_idx].mean(0) #Take the mean across video multi-clips

            assert np.all(targets[video_idx] == target), f"Error! Target don't match for the same video {video} clips."
        
            video_targets.append(target)
            video_prob_preds.append(prob_pred)

        dataset_type = dataset_type + "_consensus"

        accuracy = self._computeMetrics(video_filenames, video_targets, video_prob_preds, dataset_type, epoch)

        return accuracy

    def _runBestEpochLogic(self):

        self.best_val_acc
        # print(f'Saving the best model at epoch {epoch} with mIoU value {best_avg_mIoU}')
        # # saveModel(model, f'best_model.pth')
        # saveCheckpoint(f'best_model.pth', model, optimizer, scheduler, epoch, best_avg_mIoU)

        # is_best = True
        # #Save the best val predictions
        # if is_best:
        #     pred_path = os.path.join(constants.analysis_dir_path, "val")
        #     best_pred_path = os.path.join(constants.analysis_dir_path, "best_val")
        #     utils.removeDir(best_pred_path)
        #     os.system(f"cp -r {pred_path} {best_pred_path}")

        #     #Run evaluation on heldout test set
        #     evaluate_batch(args, model, device, heldout_test_dataloader, use_rf, epoch = epoch, 
        #             criterion = criterion, contrastiveCriterion = contrastiveCriterion, writer = writer, criterion2 = criterion2, prefix = 'test')


    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_start(trainer, pl_module)

        self.train_filenames = []
        self.train_targets = []
        self.train_prob_preds = []

    # def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused) -> None:
    #     super().on_train_epoch_end(trainer, pl_module, unused=unused)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        
        epoch = trainer.current_epoch

        accuracy = self._computeMetrics(self.train_filenames, self.train_targets, self.train_prob_preds, "Train", epoch)

        con_accuracy = self._computeConsensusMetrics(self.train_filenames, self.train_targets, self.train_prob_preds, "Train", epoch)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_start(trainer, pl_module)

        self.val_filenames = []
        self.val_targets = []
        self.val_prob_preds = []

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch

        accuracy = self._computeMetrics(self.val_filenames, self.val_targets, self.val_prob_preds, "Val", epoch)

        con_accuracy = self._computeConsensusMetrics(self.val_filenames, self.val_targets, self.val_prob_preds, "Val", epoch)

        if accuracy >= self.best_val_acc:
            self.best_val_acc = accuracy

            self._runBestEpochLogic()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_start(trainer, pl_module)

        self.test_filenames = []
        self.test_targets = []
        self.test_prob_preds = []

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch

        accuracy = self._computeMetrics(self.test_filenames, self.test_targets, self.test_prob_preds, "Test", epoch)

        con_accuracy = self._computeConsensusMetrics(self.test_filenames, self.test_targets, self.test_prob_preds, "Test", epoch)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
 
        filenames, clips, targets = batch
        preds = outputs["pred_cls"][:, :4]

        self.train_filenames.extend(filenames)
        self.train_targets.extend(targets["lb_cls"].detach().cpu().numpy())
        self.train_prob_preds.extend(preds.detach().cpu().numpy())

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        filenames, clips, targets = batch
        preds = outputs["pred_cls"][:, :4]

        self.val_filenames.extend(filenames)
        self.val_targets.extend(targets["lb_cls"].detach().cpu().numpy())
        self.val_prob_preds.extend(preds.detach().cpu().numpy())
        # self.val_prob_preds.append(preds.detach().cpu().numpy())
    
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
        filenames, clips, targets = batch
        preds = outputs["pred_cls"][:, :4]

        self.test_filenames.extend(filenames)
        self.test_targets.extend(targets["lb_cls"].detach().cpu().numpy())
        self.test_prob_preds.extend(preds.detach().cpu().numpy())
       




class SaveMetricsDataCallback(pl_callbacks.Callback):

    def __init__(self, cfg, path, dataset_type) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.path = path

        self.dataset_type = dataset_type

        
        self.test_filenames = []

        self.test_targets = []

        self.test_prob_preds = []

        
    def _saveMetricsData(self, filenames, targets, prob_preds, epoch):

        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            preds = torch.sigmoid(torch.tensor(prob_preds)).numpy() > 0.5
        else:
            preds = np.array(prob_preds).argmax(axis = 1)
        
        print(f"Epoch-{epoch} Saving metrics data for {self.dataset_type}")


        np.save(os.path.join(self.path, f'{self.dataset_type}_targets.npy'), targets)
        np.save(os.path.join(self.path, f'{self.dataset_type}_prob_preds.npy'), prob_preds)
        np.save(os.path.join(self.path, f'{self.dataset_type}_preds.npy'), preds)
        np.save(os.path.join(self.path, f'{self.dataset_type}_filenames.npy'), filenames)

        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            return

        np.savetxt(os.path.join(self.path, f'{self.dataset_type}_results.csv'), np.array([filenames, targets, preds]).T, fmt='%s,%s,%s', delimiter=',', header='filename, target, pred')

        


        

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_start(trainer, pl_module)

        self.test_filenames = []
        self.test_targets = []
        self.test_prob_preds = []

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch

        self._saveMetricsData(self.test_filenames, self.test_targets, self.test_prob_preds, epoch)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
        filenames, clips, targets = batch
        preds = outputs["pred_cls"]

        self.test_filenames.extend(filenames)
        self.test_targets.extend(targets["lb_cls"].detach().cpu().numpy())
        self.test_prob_preds.extend(preds.detach().cpu().numpy())
       


class ComputeSegmentationMetricsCallback(pl_callbacks.Callback):

    def __init__(self, path, num_classes, class_names, loggers) -> None:
        super().__init__()
        
        self.path = path

        self.num_classes = num_classes
        self.class_names = class_names
        self.loggers = loggers

        self.best_val_mean_IoU = 0
        
    def _computeMetrics(self, metrics, dataset_type, epoch):


        # mean_pixel_wise_acc, class_wise_mean_acc, mean_IoU, class_wise_mean_IoU = metrics.getMetrics()
        mean_pix_acc, class_pix_acc, mean_IoU, class_IoU = metrics.getMetrics()

        cls_pix_acc_dict = {}
        cls_iou_dict = {}
        for cls, acc, iou in zip(self.class_names, class_pix_acc, class_IoU):
            cls_pix_acc_dict[cls] = acc
            cls_iou_dict[cls] = iou

        print(f"Epoch-{epoch} {dataset_type} : mean_pix_acc = {mean_pix_acc}")
        print(f"Epoch-{epoch} {dataset_type} : cls_pix_acc_dict = \n {cls_pix_acc_dict}")
        print(f"Epoch-{epoch} {dataset_type} : mean_IoU = {mean_IoU}")
        print(f"Epoch-{epoch} {dataset_type} : cls_iou_dict = \n {cls_iou_dict}")

        self.loggers[0].experiment.log({f"mean_pix_acc/{dataset_type}": mean_pix_acc}) #, epoch)
        self.loggers[1].experiment.add_scalar(f"mean_pix_acc/{dataset_type}", mean_pix_acc, epoch)

        self.loggers[0].experiment.log({f"mean_IoU/{dataset_type}": mean_IoU}) #, epoch)
        self.loggers[1].experiment.add_scalar(f"mean_IoU/{dataset_type}", mean_IoU, epoch)

        self.loggers[0].experiment.log({f"cls_pix_acc_dict/{dataset_type}": cls_pix_acc_dict}) #, epoch)
        self.loggers[1].experiment.add_scalars(f"cls_pix_acc_dict/{dataset_type}", cls_pix_acc_dict, epoch)

        self.loggers[0].experiment.log({f"cls_iou_dict/{dataset_type}": cls_iou_dict}) #, epoch)
        self.loggers[1].experiment.add_scalars(f"cls_iou_dict/{dataset_type}", cls_iou_dict, epoch)

        return mean_IoU

    def _runBestEpochLogic(self):

        self.best_val_mean_IoU
  
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_start(trainer, pl_module)

        self.train_metrics = MetricEvaluation(self.num_classes, ignore_label = -1)
        

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        
        epoch = trainer.current_epoch

        mean_IoU = self._computeMetrics(self.train_metrics, "Train", epoch)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_start(trainer, pl_module)

        self.val_metrics = MetricEvaluation(self.num_classes, ignore_label = -1)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch

        mean_IoU = self._computeMetrics(self.val_metrics, "Val", epoch)
        
        if mean_IoU >= self.best_val_mean_IoU:
            self.best_val_mean_IoU = mean_IoU

            self._runBestEpochLogic()

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_start(trainer, pl_module)

        self.test_metrics = MetricEvaluation(self.num_classes, ignore_label = -1)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_test_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch

        accuracy = self._computeMetrics(self.test_metrics, "Test", epoch)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
 
        filenames, clips, targets_dict = batch

        preds = outputs["pred_seg"]
        targets = targets_dict["lb_seg"]

        preds = preds.argmax(dim = 2, keepdim = True)

        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        self.train_metrics.evaluate(preds, targets)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        filenames, clips, targets_dict = batch

        preds = outputs["pred_seg"]
        targets = targets_dict["lb_seg"]

        preds = preds.argmax(dim = 2, keepdim = True)

        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        self.val_metrics.evaluate(preds, targets)
    
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
        filenames, clips, targets_dict = batch

        preds = outputs["pred_seg"]
        targets = targets_dict["lb_seg"]

        preds = preds.argmax(dim = 2, keepdim = True)

        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        self.test_metrics.evaluate(preds, targets)
       



class LitClassifier(pl.LightningModule):
    """
    LitClassifier(
      (model): ...
    )
    """

    def __init__(self, cfg, model, learning_rate: float = 0.0001):
        super().__init__()

        self.cfg = cfg

        self.save_hyperparameters(ignore=["model"])
        
        self.model = model

    #Bugfix for pytorch-lightning issue #11471
    def log_grad_norm(self, grad_norm_dict) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        Example::

            # DEFAULT
            def log_grad_norm(self, grad_norm_dict):
                self.log_dict(grad_norm_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        """
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        filename, x, y_dict = batch
        # y_pred = self(x)
        out_dict = self(x)

        
        # loss = computeLoss(self.cfg, y_pred, y)
        loss = computeLoss(self.cfg, out_dict, y_dict)

        y = y_dict["lb_cls"]
        y_pred = out_dict["pred_cls"][:,:4]
        
        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            acc = ((torch.sigmoid(y_pred.detach()) > 0.5) == y).float().mean()
        else:
            acc = (y_pred.detach().argmax(axis = 1) == y).float().mean()

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_step=True)
        # self.log("train_performance", {"acc": acc, "recall": recall})
        
        # return loss, y_pred
        # return {"loss": loss, "y_pred": y_pred.detach()}

        for k, v in out_dict.items():
            out_dict[k] = v.detach()

        out_dict["loss"] = loss

        # out_dict["pred_cls"] = y_pred.detach()
        # out_dict["pred_seg"] = out_dict["pred_seg"].detach()

        return out_dict

    def validation_step(self, batch, batch_idx):
        filename, x, y_dict = batch
        # y_pred = self(x)
        out_dict = self(x)
        
        # loss = computeLoss(self.cfg, y_pred, y)
        loss = computeLoss(self.cfg, out_dict, y_dict)

        y = y_dict["lb_cls"]
        y_pred = out_dict["pred_cls"][:,:4]
        
        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            acc = ((torch.sigmoid(y_pred.detach()) > 0.5) == y).float().mean()
        else:
            acc = (y_pred.detach().argmax(axis = 1) == y).float().mean()
        
        self.log("val_loss", loss, on_step=True)
        self.log("val_acc", acc, on_step=True)

        # return loss, y_pred
        # return {"loss": loss, "y_pred": y_pred.detach()}

        for k, v in out_dict.items():
            out_dict[k] = v.detach()

        out_dict["loss"] = loss

        # out_dict["pred_cls"] = y_pred.detach()
        # out_dict["pred_seg"] = out_dict["pred_seg"].detach()

        return out_dict

    def test_step(self, batch, batch_idx):
        filename, x, y_dict = batch
        # y_pred = self(x)
        out_dict = self(x)
        
        # loss = computeLoss(self.cfg, y_pred, y)
        loss = computeLoss(self.cfg, out_dict, y_dict)

        y = y_dict["lb_cls"]
        y_pred = out_dict["pred_cls"][:,:4]
        
        if self.cfg.SOLVER.LOSS_FUNC == "binary_cross_entropy":
            acc = ((torch.sigmoid(y_pred.detach()) > 0.5) == y).float().mean()
        else:
            acc = (y_pred.detach().argmax(axis = 1) == y).float().mean()
        
        self.log("test_loss", loss)
        self.log("test_acc", acc, on_step=True)

        # return loss, y_pred
        # return {"loss": loss, "y_pred": y_pred.detach()}

        for k, v in out_dict.items():
            out_dict[k] = v.detach()

        out_dict["loss"] = loss

        # out_dict["pred_cls"] = y_pred.detach()
        # out_dict["pred_seg"] = out_dict["pred_seg"].detach()

        return out_dict

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     x, y = batch
    #     return self(x)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        
        def getModelPraramsToOptimize(cfg, model):
            
            if cfg.MODEL.PRETRAIN_FREEZE:
                params = filter(lambda p: p.requires_grad, model.parameters())
            else:
                params = model.parameters()

            return params

        if self.cfg.SOLVER.OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(getModelPraramsToOptimize(self.cfg, self.model), lr = self.cfg.SOLVER.INIT_LR) #, lr=self.hparams.learning_rate)
        # elif self.cfg.SOLVER.OPTIMIZER == "adam+freeze":
        #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.cfg.SOLVER.INIT_LR) #, lr=self.hparams.learning_rate)
        elif self.cfg.SOLVER.OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(getModelPraramsToOptimize(self.cfg, self.model), lr = self.cfg.SOLVER.INIT_LR) #, lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Wrong cfg.SOLVER.OPTIMIZER = {self.cfg.SOLVER.OPTIMIZER}!")

        #TODO-GRG: Need to add these values to config
        if self.cfg.SOLVER.SCHEDULER == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                 factor = 0.25, #0.75, #0.25, #0.5 
                 patience = 5, #patience=2, #patience=5, #patience=10,
                 verbose=True, threshold=1e-3, threshold_mode='rel',
                 cooldown=0, min_lr=1e-7, eps=1e-8)
        elif self.cfg.SOLVER.SCHEDULER == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0 = 5, #1, 
                    T_mult= 2, #1, 
                    eta_min= 1e-7, #0, 
                    last_epoch = -1, #self.cfg.SOLVER.EPOCHS, 
                    verbose = True
                )
        else:
            raise ValueError(f"Wrong cfg.SOLVER.OPTIMIZER = {self.cfg.SOLVER.OPTIMIZER}!")
        

        

        # return optimizer
        return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_acc",
                        # "monitor": "train_acc",
                    },
                }


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        if self.cfg.EXPERIMENT.DATASET == "LSU-Dataset":
            self.dataset = video_dataset.LSULungDataset
        elif self.cfg.EXPERIMENT.DATASET == "DARPA-Dataset":
            self.dataset = video_dataset.LSULungDataset
        elif self.cfg.EXPERIMENT.DATASET == "DARPA-Seg-Dataset":
            self.dataset = video_seg_dataset.LSULungSegDataset
        else:
            raise ValueError(f"Unsupported cfg.EXPERIMENT.DATASET = {cfg.EXPERIMENT.DATASET}!")

        self.process_dataset = self.dataset(self.cfg, dataset_type = 'process', process_data = True)

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        
    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.dataset(self.cfg, dataset_type = 'train')
        return self.train_dataset

    def train_dataloader(self):

        if self.train_dataset is None:
            self.get_train_dataset()

        return DataLoader(
                            self.train_dataset, 
                            batch_size = self.cfg.SOLVER.TRAIN_BATCH_SIZE, 
                            pin_memory = self.cfg.SYSTEM.PIN_MEMORY, 
                            num_workers = self.cfg.SYSTEM.NUM_WORKERS, 
                            shuffle = True,
                            persistent_workers = True #Ref: https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110 
                        )

    def get_val_dataset(self):
        if self.val_dataset is None:
            self.val_dataset = self.dataset(self.cfg, dataset_type = 'val')
        return self.val_dataset

    def val_dataloader(self):

        if self.val_dataset is None:
            self.get_val_dataset()
            
        return DataLoader(
                            self.val_dataset, 
                            batch_size = self.cfg.SOLVER.TEST_BATCH_SIZE, 
                            pin_memory = self.cfg.SYSTEM.PIN_MEMORY, 
                            num_workers = self.cfg.SYSTEM.NUM_WORKERS, 
                            shuffle = False,
                            persistent_workers = True
                        )

    def get_test_dataset(self):
        if self.test_dataset is None:
            self.test_dataset = self.dataset(self.cfg, dataset_type = 'test')
        return self.test_dataset

    def test_dataloader(self):

        if self.test_dataset is None:
            self.get_test_dataset()

        return DataLoader(
                            self.test_dataset, 
                            batch_size = self.cfg.SOLVER.TEST_BATCH_SIZE, 
                            pin_memory = self.cfg.SYSTEM.PIN_MEMORY, 
                            num_workers = self.cfg.SYSTEM.NUM_WORKERS, 
                            shuffle = False,
                            persistent_workers = True
                        )

    def get_dataloader(self, dataset, shuffle = False):
        return DataLoader(
                            dataset, 
                            batch_size = self.cfg.SOLVER.TEST_BATCH_SIZE, 
                            pin_memory = self.cfg.SYSTEM.PIN_MEMORY, 
                            num_workers = self.cfg.SYSTEM.NUM_WORKERS, 
                            shuffle = shuffle,
                            persistent_workers = True
                        )

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)


def cli_main(cfg, model):
    # cli = LightningCLI(LitClassifier, MyDataModule, seed_everything_default=1234, save_config_overwrite=True, run=False)
    
    net = LitClassifier(cfg, model)

    dataModule = MyDataModule(cfg)

    # #Initialize weights and biases 
    # wandb.init(project = cfg.EXPERIMENT.ANALYSIS_DIR, entity = "ggare")

    wandb_logger = pl_loggers.WandbLogger(
            save_dir = cfg.EXPERIMENT.DIR, 
            project = cfg.EXPERIMENT.NAME, 
            log_model = True, 
            group = cfg.EXPERIMENT.RUN_NAME, #
            name = cfg.EXPERIMENT.FOLD_NAME,
            mode = cfg.LOGGER.WANDB,
            # sync_tensorboard = True, #Synchronize wandb logs from tensorboard or tensorboardX and save the relevant events file. 
        )
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = cfg.EXPERIMENT.DIR, name = cfg.EXPERIMENT.NAME)
    loggers = [wandb_logger, tb_logger]


    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
            monitor = "val_acc", 
            mode = "max", 
            save_last = True,
            verbose = True,
            auto_insert_metric_name = True, #To insert the metric name & value in the filename
        )

    computeMetricsCallback = ComputeMetricsCallback(
            cfg = cfg,
            path = cfg.EXPERIMENT.DIR, 
            labels = dataModule.process_dataset.getLabels(), 
            class_names = dataModule.process_dataset.getLabelNames(),
            loggers = loggers,
        )

    callbacks = [checkpoint_callback, computeMetricsCallback]

    if cfg.CAM.ENABLE:
        videoGradCAMCallback = VideoGradCAMCallback(
                cfg = cfg,
                path = cfg.EXPERIMENT.DIR, 
                feature_module = model.base_model.layer4,
                target_layer_names = ["1"], 
                # model = model,
                model_name = cfg.EXPERIMENT.MODEL, 
                class_names = dataModule.process_dataset.getLabelNames(),
                loggers = loggers,
            )

        callbacks.append(videoGradCAMCallback)
    

    lr_monitor = pl_callbacks.LearningRateMonitor(
            # logging_interval='step', #Set to 'epoch' or 'step' to log lr of all optimizers at the same interval, set to None to log at individual interval according to the interval key of each scheduler. Defaults to None
            log_momentum = True, #To also log the momentum values of the optimizer, if the optimizer has the momentum or betas attribute. Defaults to False
        )

    callbacks.append(lr_monitor)


    if cfg.EXPERIMENT.MODEL == "tsm_seg" or "unet" in cfg.EXPERIMENT.MODEL:
    
        computeSegMetricsCallback = ComputeSegmentationMetricsCallback(
                path = cfg.EXPERIMENT.DIR, 
                num_classes = dataModule.process_dataset.getNumSegClasses(), 
                class_names = dataModule.process_dataset.getSegLabelNames(),
                loggers = loggers,
            )

        callbacks.append(computeSegMetricsCallback)

        videoSegCallback = VideoSegCallback(
                cfg = cfg,
                path = cfg.EXPERIMENT.DIR, 
                class_names = dataModule.process_dataset.getLabelNames(),
                seg_class_names = dataModule.process_dataset.getSegLabelNames(),
                loggers = loggers,
            )

        callbacks.append(videoSegCallback)


       # trainer = pl.Trainer()
    trainer = pl.Trainer(
            gpus = cfg.SYSTEM.NUM_GPUS, 
            accelerator = "gpu", 
            # distributed_backend = "dp", # "dp" caused error wherein the returned y_pred is a single val instead of array
            default_root_dir = cfg.EXPERIMENT.DIR, 
            callbacks = callbacks,
            logger = loggers,
            benchmark = True, #If true enables cudnn.benchmark. This flag is likely to increase the speed of your system if your input sizes don’t change. However, if it does, then it will likely make your system slower.
            # fast_dev_run=False, #Set to True or n for running n train and val epochs for debugging
            # overfit_batches=0.0, #Set to non zero value to use only that much of training data for train, val and test set to test overfitting 
            track_grad_norm = 2, # Track the L2 norm of the gradient
            auto_scale_batch_size = "binsearch", #False, #None | "power" | "binsearch", # If True, will initially run a batch size finder trying to find the largest batch size that fits into memory. The result will be stored in self.batch_size 
            check_val_every_n_epoch = 1, #2, #1, #Run validation for every n train epochs,
            max_epochs = cfg.SOLVER.EPOCHS,
        )
    
    # log gradients and model topology
    wandb_logger.watch(net)

    if cfg.EXPERIMENT.MODE == "Train":
        trainer.fit(net, dataModule)

        # trainer.test(net)
        trainer.test(ckpt_path = "best")

    if cfg.EXPERIMENT.MODE == "Test":
        trainer.test(net, dataModule, ckpt_path = cfg.MODEL.CHECK_POINT)


    if cfg.EXPERIMENT.MODE == "TestBest":
        trainer.test(net, dataModule, ckpt_path = "best")
    
    if cfg.EXPERIMENT.MODE == "TestLast":
        trainer.test(net, dataModule, ckpt_path = "last")

    
    if cfg.EXPERIMENT.MODE == "Test-generatePreds" or cfg.EXPERIMENT.MODE == "Train":
        saveMetricsDataCallback = SaveMetricsDataCallback(
                                            cfg = cfg,
                                            path = cfg.EXPERIMENT.DIR, 
                                            dataset_type="Train",
                                        )
        # trainer.callbacks += [saveMetricsDataCallback]
        trainer.callbacks = [computeMetricsCallback, saveMetricsDataCallback]
        # trainer.logger = [tb_logger]

        trainer.test(net, dataModule.get_dataloader(dataModule.get_train_dataset()), ckpt_path = "best")
        
        saveMetricsDataCallback.dataset_type ="Val"
        trainer.test(net, dataModule.get_dataloader(dataModule.get_val_dataset()), ckpt_path = "best")
        
        
        saveMetricsDataCallback.dataset_type ="Test"
        trainer.test(net, dataModule.get_dataloader(dataModule.get_test_dataset()), ckpt_path = "best")

    # cli = pl.Trainer(LitClassifier, MyDataModule, save_config_overwrite=True, run=False)
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])
