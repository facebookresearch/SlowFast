#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import pickle

import numpy as np

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
import torch
import math
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(test_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            metadata = metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            # Perform the forward pass.
            preds = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()

        if not cfg.VIS_MASK.ENABLE:
            # Update and log stats.
            test_meter.update_stats(preds.detach(), labels.detach(), video_idx.detach())
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    import numpy as np
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:
        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(model, cfg, use_train_input=False)

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)


        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                test_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                (
                    cfg.MODEL.NUM_CLASSES
                    if not cfg.TASK == "ssl"
                    else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM
                ),
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()

        # --- Custom CSV output for predictions (names + indices, no GT, no time fields) ---
        import re
        import csv
        import pandas as pd
        import numpy as np

        def path_only(s: str) -> str:
          """
          Accepts a line like '/path/file.mp4 214' or '/path/file.mp4'
          Returns '/path/file.mp4'
          """
          if not s:
              return s
          # split on whitespace, keep the first token
          return str(s).strip().split()[0]

        def parse_youtube_id_from_path(p: str) -> str:
            """
            Extract youtube_id from '.../<youtube_id>_000107_000117.mp4'.
            We capture everything before the last two '_######' chunks.
            """
            bn = os.path.basename(str(p))
            m = re.match(r'^(?P<ytid>.+)_(\d{6})_(\d{6})\.mp4$', bn)
            return m.group("ytid") if m else ""

        def load_class_names(label_map_csv_path: str):
            """
            Expect CSV with columns: id,name
            Returns a list where index is class id and value is class name.
            """
            if not label_map_csv_path or not os.path.isfile(label_map_csv_path):
                return None
            with open(label_map_csv_path, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                id_name = [(int(r["id"]), r["name"]) for r in rdr]
                id_name.sort(key=lambda x: x[0])
                return [name for _, name in id_name]

        def load_class_names(label_map_csv_path: str):
          """
          Expect CSV with columns: id,name
          Returns a list where index is class id and value is class name.
          """
          if not label_map_csv_path or not os.path.isfile(label_map_csv_path):
              return None
          with open(label_map_csv_path, "r", encoding="utf-8") as f:
              rdr = csv.DictReader(f)
              id_name = [(int(r["id"]), r["name"]) for r in rdr]
              id_name.sort(key=lambda x: x[0])
              return [name for _, name in id_name]

        # Load directly (no cfg dependency)
        label_map_csv = "/content/drive/MyDrive/Mae/data/kinetics_400_mapping.csv"
        class_names = load_class_names(label_map_csv)

        all_preds = test_meter.video_preds.clone().detach().cpu().numpy()
        ds = test_loader.dataset

        rows = []
        n = len(all_preds)

        # Warn if counts mismatch (often happens with failed decodes)
        if hasattr(ds, "_path_to_videos") and n != len(ds._path_to_videos):
            logger.warning(
                f"Predictions ({n}) != dataset paths ({len(ds._path_to_videos)}). "
                "Some items may have been skipped/filtered."
            )

        seen_paths = set()

        for i in range(n):
            # Prefer dataset path; otherwise blank
            #video_path = getattr(ds, "_path_to_videos", [""] * n)[i] if hasattr(ds, "_path_to_videos") else ""
            
            raw_path = getattr(ds, "_path_to_videos", [""] * n)[i] if hasattr(ds, "_path_to_videos") else ""
            video_path = path_only(raw_path)
            if not video_path:
                # If for some reason the dataset path is missing, skip — we need a key
                continue

            # Dedup by full path (since you don't want time fields)
            if video_path in seen_paths:
                continue
            seen_paths.add(video_path)

            youtube_id = ""
            # If dataset provides it, use it; else parse from filename
            if hasattr(ds, "_youtube_ids"):
                try:
                    youtube_id = ds._youtube_ids[i]
                except Exception:
                    youtube_id = ""
            if not youtube_id:
                youtube_id = parse_youtube_id_from_path(video_path)

            pred_scores = all_preds[i]
            top5_idx = np.argsort(pred_scores)[::-1][:5].astype(int)
            top1_idx = int(top5_idx[0])

            if class_names is not None and top1_idx < len(class_names):
                top1_lbl = class_names[top1_idx]
                top5_lbls = [class_names[j] if j < len(class_names) else str(j) for j in top5_idx]
            else:
                # Fallback to indices if mapping missing
                top1_lbl = str(top1_idx)
                top5_lbls = [str(j) for j in top5_idx]

            rows.append({
                "youtube_id": youtube_id,
                "video_path": str(video_path),
                "pred_top1": top1_lbl,
                "pred_top1_idx": top1_idx,                         # keep for reproducibility (optional)
                "pred_top5": "|".join(top5_lbls),
                "pred_top5_idx": "|".join(map(str, top5_idx)),     # keep for reproducibility (optional)
            })

        # Diagnostics: what the dataset intended to load vs what got predictions
        all_paths = []
        if hasattr(test_loader.dataset, "_path_to_videos"):
            #all_paths = [str(p) for p in test_loader.dataset._path_to_videos]
            all_paths = [path_only(p) for p in test_loader.dataset._path_to_videos]        
        else:
            # if your dataset uses a different attribute, add it here
            pass

        pred_paths = [r["video_path"] for r in rows]  # rows you’re about to write
        missing = sorted(set(all_paths) - set(pred_paths))

        logger.info(f"[DIAG] dataset listed: {len(all_paths)} paths; predicted: {len(pred_paths)}; missing: {len(missing)}")
        if missing:
            miss_csv = os.path.join(cfg.OUTPUT_DIR, "missing_predictions.csv")
            import pandas as pd
            pd.DataFrame({"video_path": list(missing)}).to_csv(miss_csv, index=False)
            logger.info(f"[DIAG] wrote missing predictions list to: {miss_csv}")

        output_csv = getattr(cfg.TEST, 'SAVE_RESULTS_CSV', None)
        if output_csv:
            output_csv_path = os.path.join(cfg.OUTPUT_DIR, output_csv)
            pd.DataFrame(rows).to_csv(output_csv_path, index=False)
            logger.info(f"Saved predictions to CSV: {output_csv_path}")


    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        logger.info(
            "Finalized testing with {} temporal clips and {} spatial crops".format(
                view, cfg.TEST.NUM_SPATIAL_CROPS
            )
        )
        result_string_views += "_{}a{}" "".format(view, test_meter.stats["top1_acc"])

        result_string = (
            "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}" "".format(
                params / 1e6,
                flops,
                view,
                test_meter.stats["top1_acc"],
                test_meter.stats["top5_acc"],
                misc.gpu_mem_usage(),
                flops,
            )
        )

        logger.info("{}".format(result_string))
    logger.info("{}".format(result_string_views))
    return result_string + " \n " + result_string_views
