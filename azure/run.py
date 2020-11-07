import argparse
import sys
import os
import copy
import datetime
import socket
import cv2
import torchvision

from common.utils.yamlConfig import YamlConfig
from common.utils.logger import CreateLogger

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from slowfast.utils.misc import launch_job
from slowfast.tools.train_net import Trainer
from slowfast.tools.onnxUtils import OnnxUtils
from slowfast.utils.parser import load_config

from common.utils.pathUtils import createFullPathTree, ensureDir
from common.utils.tempDir import TempDir

'''
Standard azure ML entry point file

July 2020
Michael Revow
'''

def ParseArgs():
  parser = argparse.ArgumentParser("videoClass")

  parser.add_argument("--root_dir", type=str, help="Experiment root directory",)
  parser.add_argument("--config_root", type=str,
                      help="Configuration directory")
  parser.add_argument("--config_files", nargs='+', help="Configuration File")
  parser.add_argument("--output_dir", type=str, help="Output directory")
  parser.add_argument("--input_file_storage", type=str, help="File storage director",)
  parser.add_argument("--log_title", type=str, help="Run title", default='None')
  parser.add_argument("--cfg_file", type=str, help="", default=None)
  parser.add_argument(
      "--SHARD_ID",
      help="The shard id of current node, Starts from 0 to NUM_SHARDS - 1",
      default=None,
      type=int,
  )
  parser.add_argument(
      "--NUM_SHARDS",
      help="Number of shards using by the job",
      default=None,
      type=int,
  )
  parser.add_argument(
      "opts",
      help="See slowfast/config/defaults.py for all options",
      default=None,
      nargs=argparse.REMAINDER,
  )
  args = parser.parse_args()
  return args

def updatePaths(args):
  '''
  Update the data paths to take into account root_dir
  After this update teh data path are absolute and we no longer need to use root_dir or output_dir
  '''
  args.DATA['PATH_TO_DATA_DIRS'] = [os.path.join(args.root_dir,d) for d in  args.DATA['PATH_TO_DATA_DIRS']]
  args.DATA['PATH_PREFIXS'] = [os.path.join(args.root_dir, d) for d in  args.DATA['PATH_PREFIXS']]
  args.OUTPUT_DIR = args.output_dir

  if args.TRAIN.get('CHECKPOINT_FILE_PATH', None) is not None:
    args.TRAIN['CHECKPOINT_FILE_PATH'] = os.path.join(args.root_dir, args.TRAIN['CHECKPOINT_FILE_PATH'])

  return args

def main():
  argsOrig = ParseArgs()
  host_name = socket.gethostname() 
  host_ip = socket.gethostbyname(host_name)        
  print("Starting on host {} host_ip {}".format(host_name, host_ip))
  for config_file in argsOrig.config_files:
    args = copy.deepcopy(argsOrig)
    args.config_file = config_file  
    with YamlConfig(args, now='') as config:
      args = config.ApplyConfigFile(args)
      args = updatePaths(args)

      with TempDir(baseDir=args.output_dir, deleteOnExit=True) as tmp:
        # Convert args to cfg format
        tmpFile = createFullPathTree(tmp.tempDir, 'cfgTmp')
        args.cfg_file = tmpFile
        config.SaveConfig(file=tmpFile)
        cfg = load_config(args)

      with CreateLogger(args, logger_type=args.logger_type) as logger:
        logger.log_value('title', args.log_title, 'Run Title entered when job started')
        logger.info("Starting on host {} host_ip {}".format(host_name, host_ip))
        logger.info("cv2 version {}".format(cv2.__version__))
        logger.info("torch version {}".format(torch.__version__))
        logger.info("Cuda enabled {} num GPU {}".format(torch.cuda.is_available(), torch.cuda.device_count()))
        logger.info("Torchvision version {}".format(torchvision.__version__))

        logger.info(config.ReportConfig())
        args.master_addr = host_ip if cfg.NUM_SHARDS <= 1  or cfg.SHARD_ID == 0 else args.master_addr
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = str(args.master_port)
        os.environ["WORLD_SIZE"] = str(cfg.NUM_SHARDS * cfg.NUM_GPUS)
        logger.info("MASTER_ADDR {} MASTER_PORT {} WORLD_SIZE {}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"], os.environ["WORLD_SIZE"]))
        logger.info("MASTER_ADDR {} MASTER_PORT {} ".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"], ))
        logger.info("CFG")
        logger.info(cfg)
        for op in args.operations: 
          if op.lower() =='train':
            trainer = Trainer(cfg)
            launch_job(cfg=cfg, init_method=None, func=trainer.train)
          elif op.lower() =='to_onnx':
            onnx = OnnxUtils(cfg, logger)
            onnx.saveOnnxModel()
          elif op.lower() == 'eval_onnx':
            onnx = OnnxUtils(cfg, logger)
            onnx.evalOnnx()            
          else:
            logger.info("Unrecognized option {} expect one of [train, to_onnx, eval_onnx]")  


if __name__ == "__main__":
  main()  
