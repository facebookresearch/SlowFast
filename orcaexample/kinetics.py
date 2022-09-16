import argparse
import os

from slowfast.models import build_model
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
from slowfast.datasets import loader
from slowfast.utils.parser import load_config

from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca.learn.trigger import EveryEpoch
from bigdl.orca import init_orca_context, stop_orca_context


parser = argparse.ArgumentParser(description='PyTorch Kinectics Example')
parser.add_argument('--cluster_mode', type=str, default="local",
                    help='The cluster mode, such as local, yarn-client, yarn-cluster, spark-submit or k8s.')
parser.add_argument('--cfg_files', type=str, default="./SLOWFAST_8x8_R50.yaml",
                    help='The path to config file')
parser.add_argument('--backend', type=str, default="bigdl",
                    help='The backend of PyTorch Estimator; bigdl, ray, and spark are supported')
parser.add_argument("--executor_memory", type=str, default="8g", help="executor memory")
parser.add_argument("--driver_memory", type=str, default="8g", help="driver memory")
parser.add_argument(
    "--opts",
    help="See slowfast/config/defaults.py for all options",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

if args.cluster_mode == "local":
    init_orca_context(memory="12g", cores=8)
elif args.cluster_mode.startswith("yarn"):
    hadoop_conf = os.environ.get("HADOOP_CONF_DIR")
    invalidInputError(
        hadoop_conf is not None,
        "Directory path to hadoop conf not found for yarn-client mode. Please "
        "set the environment variable HADOOP_CONF_DIR")
    additional = None if not os.path.exists("dataset/tiny-kinetics-400.zip") else "dataset/tiny-kinetics-400.zip#dataset"
    init_orca_context(cluster_mode="yarn-cluster", memory=args.executor_memory, driver_memory=args.driver_memory)
elif args.cluster_mode == "spark-submit":
    init_orca_context(cluster_mode="spark-submit")
else:
    invalidInputError(False,
                      "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                      " 'spark-submit', but got " + args.cluster_mode)

def reduceWrapper(func):
    def reduceIterElement(batch):
        batch=func(batch)
        print("reduce called")
        assert len(batch)==5, "Should yield inputs, labels, index, time, meta in dataloader"
        return batch[0], batch[1]
    return reduceIterElement

def train_loader_creator(config, batch_size):
    train_loader = loader.construct_loader(config, "train")
    loader.shuffle_dataset(train_loader, 0)
    train_loader.collate_fn = reduceWrapper(train_loader.collate_fn)
    return train_loader

def validation_data_creator(config, batch_size):
    val_loader = loader.construct_loader(config,"val")
    val_loader.collate_fn = reduceWrapper(val_loader.collate_fn)
    return val_loader

def model_creator(config):
    return build_model(config)

def optim_creator(model, config):
    return optim.construct_optimizer(model, config)

def loss_creator(config):
    return losses.get_loss_func(config.MODEL.LOSS_FUNC)(reduction="mean")

cfg = load_config(args, args.cfg_files)

if args.backend == "bigdl":
    loss_fun = loss_creator(cfg)
    net = model_creator(cfg)
    optimizer = optim_creator(model=net, config=cfg)
    orca_estimator = Estimator.from_torch(model=net,
                                          optimizer=optimizer,
                                          loss = loss_fun,
                                          metrics=[Accuracy()],
                                          backend=args.backend,
                                          config=cfg)
    orca_estimator.fit(data=train_loader_creator(cfg, cfg.TRAIN.BATCH_SIZE),
                       validation_data=validation_data_creator(cfg,cfg.TEST.BATCH_SIZE),
                       epochs=cfg.SOLVER.MAX_EPOCH,
                       checkpoint_trigger=EveryEpoch()
                       )
    val_stats = orca_estimator.evaluate(data=validation_data_creator(cfg,cfg.TEST.BATCH_SIZE))
    print("===> Validation Complete: Top1Accuracy {}".format(val_stats["Accuracy"]))
# elif args.backend in ["ray", "spark"]:
#     orca_estimator = Estimator.from_torch(model=model_creator,
#                                           optimizer=optim_creator,
#                                           loss=loss_creator,
#                                           metrics=[Accuracy()],
#                                           backend=args.backend,
#                                           config=cfg,
#                                           model_dir=os.getcwd(),
#                                           use_tqdm=True)
#     orca_estimator.fit(data=train_loader_creator,
#                        validation_data=validation_data_creator,
#                        batch_size=cfg.TRAIN.BATCH_SIZE,
#                        epochs=cfg.SOLVER.MAX_EPOCH)
#     val_stats = orca_estimator.evaluate(data=validation_data_creator, batch_size=cfg.TEST.BATCH_SIZE)
#     print("===> Validation Complete: Top1Accuracy {}".format(val_stats["Accuracy"]))
#     orca_estimator.shutdown()
else:
    invalidInputError(False, "Only bigdl, ray, and spark are supported "
                        "as the backend, but got {}".format(args.backend))

stop_orca_context()
