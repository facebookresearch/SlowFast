'''
Convert a model to Onnx and evaluate Onnx

November 2020
'''

import numpy as np
import pandas as pd
import torch
import time
import os
import re
import psutil
import onnxruntime as ort
import matplotlib.pyplot as plt

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.utils.meters import ValMeter
import slowfast.utils.metrics as metrics
from slowfast.models import build_model
from slowfast.datasets import loader
import slowfast.models.optimizer as optim

from common.utils.pathUtils import createFullPathTree, ensureDir, addFilenameSuffix
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc


class OnnxUtils(object):

  def __init__(self, cfg, logger):
    self.cfg = cfg
    self.logger = logger
    self.onnxDevice = torch.device(self.cfg.ONNX.DEVICE)

  def getOnnxModelPath(self):
    # Create a onnx file name. Use name specified in cfg when available
    # else create a default name
    modelName = None
    if hasattr(self.cfg.ONNX, 'MODEL_NAME') and len(self.cfg.ONNX.MODEL_NAME) > 0:
      modelName = self.cfg.ONNX.MODEL_NAME
    else:
      modelName = "{}_{}_{}.onnx".format(
        self.logger.exp.name,
        self.logger.stepName(),
        self.logger.stepId()
        )
      
    onnxPathName = createFullPathTree(self.cfg.root_dir, self.cfg.ONNX.SAVE_PATH,)
    ensureDir(onnxPathName)
    onnxPathName = createFullPathTree(onnxPathName, modelName)
    return onnxPathName, modelName


  def saveOnnxModel(self):
    model = build_model(self.cfg)
    optimizer = optim.construct_optimizer(model, self.cfg)
    start_epoch = cu.load_train_checkpoint(self.cfg, model, optimizer, self.logger)

    self.cfg.TRAIN['BATCH_SIZE'] = self.cfg.ONNX.BATCH_SIZE
    dl = loader.construct_loader(self.cfg, "train")
                        
    inputs, labels, _, _ = next(iter(dl))
    if isinstance(inputs, (list,)):
      for i in range(len(inputs)):
        inputs[i].to(self.onnxDevice)

    model.to(torch.device(self.onnxDevice))
    model.eval()
    onnxPath, _ = self.getOnnxModelPath()
    

    with torch.no_grad():
        torch.onnx.export(model, 
            inputs, 
            onnxPath, 
            opset_version=self.cfg.ONNX.OPSET_VER, 
            verbose=True,
            input_names=self.cfg.ONNX.INPUT_NAMES,
            output_names=self.cfg.ONNX.OUTPUT_NAMES,
        )

    self.logger.info("Exported {}".format(onnxPath))


  def evalOnnx(self):
    for dSet in self.cfg.ONNX.DATA_SETS:
      self.evalOnnxOne(dSet)

  def plotPrCurve(self, name, nameDetail, xData, yData, xLabel='Precision', yLabel='Recall'):
    plt.close('all')
    plt.figure(figsize=(13,7))
    fig, ax = plt.subplots()
    plt.plot(xData, yData)
    title = '{} {} = {:.2f} Count {}'.format(name, nameDetail, auc( xData, yData), len(xData))
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    # if self.args.metrics_plt_pause_delay_sec > 0:
    #   plt.pause(self.args.metrics_plt_pause_delay_sec)
    self.logger.log_image(title,  plot=fig)   

  def loadOnnx(self):
    onnxPath, oFile = self.getOnnxModelPath()
    self.logger.info("Loading onnx from {} OnnxRuntime Device {} OnnxRuntime Version {}".format(onnxPath, ort.get_device(), ort.__version__))
    sess = ort.InferenceSession(onnxPath)
    inNames = [x.name for x in sess.get_inputs()]
    outNames = sess.get_outputs()[0].name
    self.logger.info("Onnx loaded inputs {} outputs {}".format(inNames, outNames))
    inNames = [x.name for x in sess.get_inputs()]
    outNames = sess.get_outputs()[0].name
    return sess, inNames, outNames

  def evalOnnxOne(self, dataSet):
    self.cfg.TRAIN['BATCH_SIZE'] = 1
    dl = loader.construct_loader(self.cfg, dataSet)
    meter = ValMeter(len(dl), self.cfg, self.logger)

    totTime = 0
    results = []
    predsAll = []
    labelsAll = []
    GB = 1024 ** 3
    meta = {'id': ['s'] * 100}
    sess, inNames, outNames = self.loadOnnx()

    for cur_iter, (frames, labels, idx, meta) in enumerate(dl):
      # if sess is None or cur_iter % 50000 == 0:
      #   if sess is not None:
      #     del sess
      #   sess, inNames, outNames = self.loadOnnx()

     
      startTime = time.time()
      meter.iter_tic()
      
      idx = torch.tensor(cur_iter)

      # assert isinstance(frames, (list,)), 'Expect a list as input'
      # assert len(inNames) == len(frames), "Num Onnx inputs {} [{}] Data has {}".format(len(inNames), inNames, len(frames))
 
      inputs = { n: d.numpy().astype(np.float32) for n, d in zip(inNames, frames)}

      pred = sess.run([outNames], inputs).pop().squeeze()

      # extract the track start and end from meta data
      # m = re.search('.*tr_([\d]+)_st_([\d]+)_end_([\d]+).pt', meta['id'][0])
      # tr, st, end = [int(x) for x in m.groups()]      
      meter.iter_toc()
      results.append((meta['id'], pred, labels))

      if np.isnan(pred).any():
        self.logger.info("NAN Iter {} Idx {}  {}  {}".format(cur_iter, idx.item(), meta, pred))
        continue

      if labels != 0.0 and labels != 1.0:
        self.logger.info("Bad Label Iter {} Idx {}  {} Label {}".format(cur_iter, idx.item(), meta, labels))
        continue

      # num_topks_correct = metrics.topks_correct(torch.tensor(pred), labels, (1, min(5, self.cfg.MODEL.NUM_CLASSES)))
      # top1_err, _ = [(1.0 - x / pred.size(0)) * 100.0 for x in num_topks_correct ]
      meter.update_stats(0, 0, 1)
      meter.update_predictions(pred, labels)
      labels = labels.numpy().astype(np.float32)[0]


      predsAll.append(pred[-1])
      labelsAll.append(labels)
      meter.log_iter_stats(0, cur_iter, predsAll, labelsAll)

      totTime += time.time() - startTime

      if len(results) % self.cfg.LOG_PERIOD == 0:
        c = len(results)
        mm = psutil.virtual_memory()
        ss = psutil.swap_memory()
        self.logger.info("{} {} eTime {:.4f} sec avg {:.4f} sec CPU {} used GB {:.3f} AVAIL {:.3f} FREE {:.3f} SWAP Tot {:.3f} SWAP USED {:.3f} SWAP Free {:.3f} ".format(
          dataSet, c, totTime, totTime / max(c, 1), psutil.cpu_percent(),
          mm.used / GB, mm.available / GB, mm.free / GB, 
          ss.total / GB, ss.used/GB, ss.free/GB, ))

    c = len(results)
    eTime = totTime / max(c, 1)
    self.logger.info("{} {} eTime {} sec avg {} sec".format(dataSet, c, totTime, eTime))
    meter.log_epoch_stats(0, predsAll, labelsAll)
    
    if len(self.cfg.ONNX.SAVE_PREDS_PATH) > 0:
      df = pd.DataFrame(results, columns=['clip', 'pred', 'label'])
      dfPath = createFullPathTree(self.cfg.OUTPUT_DIR, addFilenameSuffix(self.cfg.ONNX.SAVE_PREDS_PATH, dataSet))
      ensureDir(os.path.dirname(dfPath))
      self.logger.info("Saving predictions results {}".format(dfPath))
      df.to_pickle(dfPath)

    if self.cfg.ONNX.PREC_RECALL_PLOT:
      precision, recall, thresholds = precision_recall_curve(labelsAll, predsAll)
      self.plotPrCurve(dataSet, 'PR AUC',  recall, precision, xLabel='Recall', yLabel='Precision',)
    if self.cfg.ONNX.ROC_PLOT:
      fpr, tpr, rocThresholds = roc_curve(labelsAll, predsAll)
      self.plotPrCurve(dataSet, 'ROC AUC', fpr, tpr, xLabel='FPR', yLabel='TPR')