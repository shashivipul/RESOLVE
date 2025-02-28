
import os
import numpy as np
from datetime import datetime
import argparse
from Utils import *
from model import *
from load_data import *
from trainer import Trainer


        
def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()
######################## Model parameters ########################f
home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=42, type=int, help='seed value')

# 1. self_supervised pre_train; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='pre_train', type=str,
                    help='pre_train, fine_tune_test')

parser.add_argument('dataset', default='ABIDE', type=str,
                    help='Dataset of choice: MDD, ABIDE')

parser.add_argument('--logs_save_dir', default='../experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args, unknown = parser.parse_known_args()

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

dataset = args.dataset
experiment_description = str(dataset) 


method = 'GCN'
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)
exec(f'from config_files.{dataset}_Configs import Config as Configs')
configs = Configs()

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}_2layergcn")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {dataset}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
dataset_name = dataset
atlas_name = "AAL"

sourcedata_path = f"../{dataset_name}/{atlas_name}/X.npz"
subset = True  # if subset= true, use a subset for debugging.

"""Here are two models, one basemodel, another is temporal contrastive model"""
GCN_model = GCN(configs).to(device)
classifier = TargetClassifier(configs).to(device)
temporal_contr_model = None


if training_mode != "pretrain":
    #load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
    f"pre_train_seed_{SEED}_2layergcn", "saved_models"))
    print("The loading file path", load_from)
    chkpoint = torch.load(os.path.join(load_from, "ckp_last_NYU.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    GCN_model.load_state_dict(pretrained_dict)
    #GCN_model = GCN(configs).to(device)

model_optimizer = torch.optim.Adam(GCN_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Trainers
Trainer(GCN_model, model_optimizer, classifier, classifier_optimizer, sourcedata_path, device,
        configs, experiment_log_dir, training_mode)

logger.debug(f"Training time is : {datetime.now()-start_time}")
