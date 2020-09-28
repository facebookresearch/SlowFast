import argparse
import yaml
import copy
import os
import re

from common.utils.yamlConfig import YamlConfig
from common.utils.logger import CreateLogger
from common.azure.azureSubmitter import AzureSubmitter
from common.utils.pathUtils import  createFullPathTree, addFilenameSuffix
from common.utils.tempDir import TempDir

'''
azureSubmitMulti.py - 
Submits multiple jobs to azure to spread a job avoer multiple shards
Taks a single azure_config_file and updates the shard ID for each job
submitted

September 2020
Michael Revow
'''

def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config_root',
      type=str,
      default="./",
      help='Root directory for config files'
  )

  parser.add_argument(
      '--config_file',
      type=str,
      default='azureSubmit_gpu.yaml',
      help='Submit config file to load'
  )
  parser.add_argument(
    '--step_name',
    type=str,
    default='step',
    help='Step name '
  )
  parser.add_argument(
    '--do_canary',
    action='store_true',
    default=False,
    help='Create a canary run'
  )

  parser.add_argument(
      '--azure_config_files',
      nargs='*',
      help='Config file(s) on azure cluster Specify at least one'
  )
  parser.add_argument(
      '--exp',
      default='tmp',
      help='Azure experiment names'
  )
  parser.add_argument(
      '--title',
      type=str,
      default='No title provided',
      help='Title for run'
  )
  parser.add_argument(
      '--aml_compute_target',
      type=str,
      default=None,
      help='Compute cluster to use'
  )
  parser.add_argument(
    '--dry_run',
    action='store_true',
    default=False,
    help='Dry run does not submit jobs just prints what will be submitted'
  )

  args = parser.parse_args()
  return args


def CreateConfig(inpt, outpt, shardId, masterIp):
  with open(outpt, 'w') as fpOut:
    with open(inpt, 'r') as fpIn:
      # Replace any existing config parameters with updated values
      for line in fpIn.readlines():
        line = line.rstrip()
        m = re.match('^SHARD_ID', line)
        if m is not None:
          line = 'SHARD_ID: {}'.format(shardId)

        m = re.match('^master_addr', line)
        if masterIp is not None and m is not None:
          line = "master_addr: {}".format(masterIp)
        print(line, file=fpOut)

  return [outpt]

def main():
  argsOrig = parseArgs()
  config = YamlConfig(argsOrig)
  argsOrig = config.ApplyConfigFile(argsOrig)
  inpt = argsOrig.azure_config_files[0]
  with open(inpt, "r") as fp:
    azureConfig = yaml.load(fp, Loader=yaml.FullLoader)
    numShards = azureConfig.get('NUM_SHARDS', 1)

  with CreateLogger(argsOrig, logger_type='passthrough') as logger:
    logger.info(config.ReportConfig())
    logger.info("Submiting to {} shards".format(numShards))
    masterIp = None

    with TempDir(baseDir=os.path.dirname("."), tempSubDir='submitTmp', deleteOnExit=False) as tt: 
      for shardId in range(0, numShards):
        args = copy.deepcopy(argsOrig)
        output = createFullPathTree(tt.tempDir, addFilenameSuffix(inpt, "_shard_{}".format(shardId)))
        # output = addFilenameSuffix(inpt, "_shard_{}".format(shardId))

        args.azure_config_files = CreateConfig(inpt, output, shardId, masterIp)
        args.step_name = '{}_{}'.format(args.step_name, shardId)
        args.title = 'Start at {} {}'.format(shardId, args.title)
        logger.info("Experiment {}  azure_config_files {} step_name {} title {}".format(args.exp, args.azure_config_files, args.step_name, args.title))

        if not args.dry_run:
          with AzureSubmitter(args, logger) as submitter:
            submitter.run()

        if numShards > 1 and shardId == 0:
          masterIp = input("Enter IP address for master shard ")
          logger.info("Master IP {}".format(masterIp))


if __name__ == "__main__":
  main()
