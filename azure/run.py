import argparse
import sys
import os
import copy
import datetime

from common.utils.yamlConfig import YamlConfig
from common.utils.logger import CreateLogger

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from slowfast.utils.misc import launch_job
from slowfast.tools.train_net import Trainer
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
      "--shard_id",
      help="The shard id of current node, Starts from 0 to num_shards - 1",
      default=0,
      type=int,
  )
  parser.add_argument(
      "--num_shards",
      help="Number of shards using by the job",
      default=1,
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
  args.DATA['PATH_TO_DATA_DIR'] = os.path.join(args.root_dir, args.DATA['PATH_TO_DATA_DIR'])
  args.DATA['PATH_PREFIX'] = os.path.join(args.root_dir, args.DATA['PATH_PREFIX'])
  args.OUTPUT_DIR = args.output_dir

  return args

def main():
  argsOrig = ParseArgs()
  for config_file in argsOrig.config_files:
    args = copy.deepcopy(argsOrig)
    args.config_file = config_file  
    with YamlConfig(args, now='') as config:
      args = config.ApplyConfigFile(args)
      args = updatePaths(args)

      with TempDir(baseDir=args.output_dir, deleteOnExit=True) as tmp:
        tmpFile = createFullPathTree(tmp.tempDir, 'cfgTmp')
        args.cfg_file = tmpFile
        config.SaveConfig(file=tmpFile)
        cfg = load_config(args)

      with CreateLogger(args, logger_type=args.logger_type) as logger:
        logger.log_value('title', args.log_title, 'Run Title entered when job started')
        # logger.info(config.ReportConfig())
        logger.info("CFG")
        logger.info(cfg)
        ite = 0
        loss = 1.1
        trainer = Trainer(cfg)
        launch_job(cfg=cfg, init_method=args.init_method, func=trainer.train)

if __name__ == "__main__":
  main()  