from src.inference import complete_solving_accelerate_v3, BN_eval_MNIST, BN_eval_CIFAR10, load_cnf_dnf_block
import os
from copy import copy, deepcopy
import time
from src.inference import get_mapping_filter, Binarize01Act, InputQuantizer, load_data, \
    infer_normal_withPYTHON, get_refs_all2, get_dictionnary_ref, \
    get_refs_all_cnn
import torch
import argparse
from tqdm import tqdm
from config.config import Config, two_args_str_float, transform_input_filters_multiple, \
    transform_input_filters2, transform_input_thr
from config.config import str2bool, two_args_str_int, str2list, \
    transform_input_filters, transform_input_lr
import torch.nn as nn
import numpy as np


from src.inference import load_TT_TTnoise

import random

config_general = Config(path="config/")
if config_general.dataset=="CIFAR10":
    config = Config(path="config/cifar10/")
elif config_general.dataset=="MNIST":
    config = Config(path="config/mnist/")
else:
    raise 'PB'


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default=config_general.dataset)

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, choices=["cuda", "cpu"])
parser.add_argument("--device_ids", default=config.general.device_ids, type=str2list)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--num_workers", default=config.general.num_workers, type=int)
parser.add_argument("--quant_step", default=config.model.quant_step, type=two_args_str_float)
parser.add_argument("--famille", default=config.model.famille)
parser.add_argument("--cbd", default=config.model.cbd)
parser.add_argument("--first_layer", default=config.model.first_layer, choices=["float", "bin"])
parser.add_argument("--preprocessing_CNN", default=config.model.preprocessing_CNN, type=transform_input_filters)
parser.add_argument("--g_remove_last_bn", default=config.model.g_remove_last_bn)
parser.add_argument("--type_blocks", default=config.model.type_blocks, type=transform_input_filters2)
parser.add_argument("--last_layer", default=config.model.last_layer, choices=["float", "bin"])
parser.add_argument("--Blocks_filters_output", default=config.model.Blocks_filters_output, type=transform_input_filters)
parser.add_argument("--Blocks_amplifications", default=config.model.Blocks_amplifications, type=transform_input_filters)
parser.add_argument("--Blocks_strides", default=config.model.Blocks_strides, type=transform_input_filters)
parser.add_argument("--type_first_layer_block", default=config.model.type_first_layer_block, choices=["float", "bin"])
parser.add_argument("--kernel_size_per_block", default=config.model.kernel_size_per_block, type=transform_input_filters)
parser.add_argument("--groups_per_block", default=config.model.groups_per_block, type=transform_input_filters)
parser.add_argument("--padding_per_block", default=config.model.padding_per_block, type=transform_input_filters)
parser.add_argument("--kernel_size_per_block_multihead", default=config.model.kernel_size_per_block_multihead, type=transform_input_filters_multiple)
parser.add_argument("--groups_per_block_multihead", default=config.model.groups_per_block_multihead, type=transform_input_filters_multiple)
parser.add_argument("--paddings_per_block_multihead", default=config.model.paddings_per_block_multihead, type=transform_input_filters_multiple)

parser.add_argument("--adv_epsilon", default=config.train.adv_epsilon)
parser.add_argument("--adv_step", default=config.train.adv_step)
parser.add_argument("--n_epoch", default=config.train.n_epoch, type=two_args_str_int)
parser.add_argument("--lr", default=config.train.lr, type=transform_input_lr)

parser.add_argument("--batch_size_test", default=config.eval.batch_size_test, type=two_args_str_int)
parser.add_argument("--jeudelavie", default=config.eval.jeudelavie, type=str2bool)
parser.add_argument("--pruning", default=config.eval.pruning, type=str2bool)
parser.add_argument("--coef_mul", default=config.eval.coef_mul, type=two_args_str_int)
parser.add_argument("--path_save_model", default=config.eval.path_load_model, type=two_args_str_int)

parser.add_argument("--Transform_normal_model", default=config.NN2TT.Transform_normal_model, type=str2bool)
parser.add_argument("--Transform_pruned_model", default=config.NN2TT.Transform_pruned_model, type=str2bool)
parser.add_argument("--Transform_normal_model_with_filtering", default=config.NN2TT.Transform_normal_model_with_filtering, type=str2bool)
parser.add_argument("--Transform_pruned_model_with_filtering", default=config.NN2TT.Transform_pruned_model_with_filtering, type=str2bool)
parser.add_argument("--filter_occurence", default=config.NN2TT.filter_occurence, type=two_args_str_int)
parser.add_argument("--block_occurence", default=config.NN2TT.block_occurence, type=two_args_str_int)

parser.add_argument("--Add_noise", default=config.NN2TT.Add_noise, type=str2bool)
parser.add_argument("--proportion", default=config.NN2TT.proportion, type=two_args_str_float)
parser.add_argument("--proba", default=config.NN2TT.proba, type=two_args_str_float)

parser.add_argument("--model_to_eval", default=config.verify.model_to_eval)
parser.add_argument("--type_verification", default=config.verify.type_verification)
parser.add_argument("--mode_verification", default=config.verify.mode_verification)
parser.add_argument("--ratio_incomplet", default=config.verify.ratio_incomplet, type=two_args_str_float)
parser.add_argument("--attack_eps", default=config.verify.attack_eps, type=two_args_str_float)
parser.add_argument("--coef_multiplicateur_data", default=config.verify.coef_multiplicateur_data, type=two_args_str_int)
parser.add_argument("--offset", default=config.verify.offset, type=two_args_str_int)
parser.add_argument("--encoding_type", default=config.verify.encoding_type, type=two_args_str_int)
parser.add_argument("--sat_solver", default=config.verify.sat_solver)
parser.add_argument("--time_out", default=config.verify.time_out, type=two_args_str_int)
parser.add_argument("--thr_bin_act", default=config.model.thr_bin_act, type=transform_input_thr)
parser.add_argument("--thr_bin_act_test", default=config.eval.thr_bin_act_test, type=transform_input_thr)
parser.add_argument("--method_verify_incomplete", default=config.verify.method_verify_incomplete,choices=["DP", "formula"])

parser.add_argument("--with_contradiction", default=config.NN2TT.with_contradiction, type=str2bool)

args = parser.parse_args()
args.preprocessing_BN = 1
args.batch_size_test=1000

args.path_save_model = args.path_save_model+"/"
args.path_load_model = args.path_save_model+"/"
device = "cpu" #torch.device("cuda:" + str(args.device_ids[0]) if torch.cuda.is_available() else "cpu")
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



if config_general.dataset=="CIFAR10":
    args.attack_eps = args.attack_eps / 255

print(args)


print()
print("-"*100)
print()
print("START EVALUATION ")
print()

print(args.path_save_model)
dataloaders, testset, nclass = load_data(args)

all_CNFG = {i: {j: [] for j in range(10)} for i in range(10)}

liste_fonctions1 = []
liste_fonctions1.append(InputQuantizer(args.quant_step))
preprocessing_inpQ = nn.Sequential(*liste_fonctions1).eval()
print(preprocessing_inpQ)

#Preprocessing
liste_fonctions = []
liste_fonctions2 = []
liste_fonctions_P = []
liste_fonctions_N = []
liste_fonctions.append(InputQuantizer(args.quant_step))
liste_fonctions2.append(InputQuantizer(args.quant_step))
liste_fonctions_P.append(InputQuantizer(args.quant_step))
liste_fonctions_N.append(InputQuantizer(args.quant_step))


scale = np.loadtxt(args.path_load_model+"/preprocessing_BN_scale.txt")
bias = np.loadtxt(args.path_load_model+"/preprocessing_BN_bias.txt")
print(scale.shape, bias.shape)
if config_general.dataset == "MNIST":
    liste_fonctions.append(BN_eval_MNIST(np.array([scale]), np.array([bias])).to(device))
else:
    liste_fonctions.append(BN_eval_CIFAR10(np.array([scale]), np.array([bias])).to(device))
#liste_fonctions.append(BN_eval(np.array([scale]),np.array([bias])).to(device))
if os.path.isfile(args.path_load_model+"/preprocessing_BN_scale_2.txt"):
    scale2 = np.loadtxt(args.path_load_model + "/preprocessing_BN_scale_2.txt")
    bias2 = np.loadtxt(args.path_load_model + "/preprocessing_BN_bias_2.txt")
    if config_general.dataset == "MNIST":
        liste_fonctions.append(BN_eval_MNIST(np.array([scale2]), np.array([bias2])).to(device))
    else:
        liste_fonctions.append(BN_eval_CIFAR10(np.array([scale2]), np.array([bias2])).to(device))



#ok
#count, putawayliteral = load_cnf_dnf_block_count(args)
putawayliteral = []
print(putawayliteral)
#putawayliteral = [(0, 2, 5), (0, 4, 5), (0, 4, 8), (0, 6, 0), (0, 6, 5), (0, 6, 8), (0, 7, 0), (0, 7, 1), (0, 7, 5), (0, 7, 6), (0, 7, 7), (0, 9, 1), (0, 9, 2), (0, 9, 5), (0, 9, 6), (0, 9, 8), (0, 10, 0), (0, 10, 1), (0, 10, 2), (0, 10, 3), (0, 12, 2), (0, 12, 5), (0, 12, 7), (0, 13, 0), (0, 13, 1), (0, 13, 3), (0, 13, 8), (0, 14, 7), (0, 16, 7), (0, 18, 7), (0, 19, 1), (0, 19, 2), (0, 19, 3), (0, 19, 8), (0, 22, 0), (0, 22, 2), (0, 22, 6) , (0, 22, 7), (0, 22, 8), (0, 24, 5), (0, 25, 3), (0, 25, 4), (0, 25, 5), (0, 25, 6), (0, 25, 7), (0, 25, 8), (0, 26, 1), (0, 26, 5), (0, 26, 6), (0, 26, 8), (0, 27, 0), (0, 27, 5), (0, 27, 6), (0, 27, 8), (0, 30, 1), (0, 30, 8), (0, 31, 0), (0, 31, 1), (0, 31, 2), (0, 31, 3), (0, 31, 4), (0, 31, 5), (0, 31, 6), (0, 31, 7), (0, 31, 8), (0, 32, 0), (0, 32, 1), (0, 32, 2), (0, 32, 3), (0, 32, 4), (0, 32, 6), (0, 32, 7), (0, 32, 8), (0, 33, 2), (0, 33, 8), (0, 34, 7), (0, 35, 0), (0, 35, 1), (0, 35, 2), (0, 35, 6), (0, 35, 7), (0, 36, 2), (0, 36, 3), (0, 36, 5), (0, 36, 6), (0, 39, 0), (0, 39, 6), (0, 39, 7), (0, 39, 8), (0, 40, 3), (0, 40, 5), (0, 43, 0), (0, 43, 1), (0, 43, 2), (0, 43, 3), (0, 43, 4), (0, 43, 5), (0, 44, 7), (0, 45, 1), (0, 45, 2), (0, 45, 4), (0, 45, 7), (1, 0, 1), (1, 0, 2), (1, 1, 2), (1, 1, 6), (1, 1, 8), (1, 2, 7), (1, 2, 8), (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 3, 6), (1, 3, 8), (1, 5, 4), (1, 5, 6), (1, 5, 8), (1, 7, 5), (1, 8, 2), (1, 8, 7), (1, 8, 8), (1, 11, 0), (1, 11, 3), (1, 11, 7), (1, 15, 1), (1, 15, 2), (1, 15, 5), (1, 15, 6), (1, 15, 7), (1, 15, 8), (1, 22, 5), (1, 22, 6), (1, 22, 7), (1, 22, 8), (1, 24, 6), (1, 28, 6), (1, 29, 0), (1, 29, 8), (1, 37, 0), (1, 37, 1), (1, 37, 2), (1, 37, 3), (1, 37, 4), (1, 37, 6), (1, 37, 7), (1, 37, 8), (1, 38, 6), (1, 38, 8), (1, 39, 1), (1, 39, 6), (1, 40, 2), (1, 40, 5), (1, 40, 8), (1, 41, 8)]
#putawayliteral = [(0, 2, 3), (0, 2, 5), (0, 3, 0), (0, 3, 1), (0, 3, 5), (0, 3, 8), (0, 4, 5), (0, 4, 8), (0, 6, 3), (0, 6, 5), (0, 7, 0), (0, 7, 2), (0, 7, 4), (0, 8, 0), (0, 8, 5), (0, 8, 7), (0, 8, 8), (0, 9, 1), (0, 9, 3), (0, 10, 3), (0, 10, 6), (0, 11, 3), (0, 12, 3), (0, 12, 4), (0, 13, 4), (0, 13, 6), (0, 14, 3), (0, 14, 7), (0, 15, 7), (0, 16, 7), (0, 17, 1), (0, 18, 7), (0, 19, 1), (0, 19, 4), (0, 20, 1), (0, 20, 5), (0, 21, 3), (0, 22, 2), (0, 22, 7), (0, 22, 8), (0, 23, 1), (0, 24, 4), (0, 24, 5), (0, 25, 2), (0, 25, 6), (0, 26, 4), (0, 26, 8), (0, 27, 3), (0, 28, 1), (0, 28, 3), (0, 28, 4), (0, 29, 5), (0, 29, 8), (0, 30, 8), (0, 31, 1), (0, 31, 2), (0, 31, 8), (0, 33, 1), (0, 33, 2), (0, 34, 1), (0, 34, 7), (0, 34, 8), (0, 35, 0), (0, 36, 3), (0, 36, 4), (0, 36, 6), (0, 37, 3), (0, 37, 5), (0, 38, 5), (0, 39, 0), (0, 39, 1), (0, 39, 4), (0, 40, 0), (0, 40, 2), (0, 40, 3), (0, 41, 5), (0, 41, 6), (0, 41, 7), (0, 42, 4), (0, 43, 0), (0, 43, 1), (0, 43, 2), (0, 43, 3), (0, 43, 4), (0, 44, 1), (0, 44, 2), (0, 44, 4), (0, 44, 5), (0, 45, 1), (0, 45, 4), (0, 45, 7), (0, 46, 5), (0, 46, 6), (0, 46, 7), (0, 47, 4), (0, 47, 5), (0, 47, 7), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 1, 7), (1, 2, 0), (1, 2, 4), (1, 2, 8), (1, 3, 0), (1, 3, 2), (1, 3, 6), (1, 3, 8), (1, 4, 3), (1, 4, 7), (1, 5, 5), (1, 5, 8), (1, 6, 0), (1, 6, 8), (1, 7, 0), (1, 7, 4), (1, 8, 3), (1, 8, 5), (1, 9, 2), (1, 9, 6), (1, 10, 0), (1, 10, 4), (1, 10, 6), (1, 11, 0), (1, 11, 3), (1, 11, 7), (1, 11, 8), (1, 12, 3), (1, 12, 8), (1, 15, 2), (1, 15, 6), (1, 16, 8), (1, 18, 0), (1, 18, 1), (1, 18, 5), (1, 18, 6), (1, 19, 0), (1, 19, 2), (1, 19, 4), (1, 20, 2), (1, 21, 2), (1, 21, 5), (1, 21, 8), (1, 22, 5), (1, 22, 7), (1, 22, 8), (1, 23, 0), (1, 23, 1), (1, 24, 1), (1, 25, 1), (1, 25, 2), (1, 25, 3), (1, 25, 8), (1, 26, 3), (1, 26, 8), (1, 27, 1), (1, 28, 5), (1, 28, 6), (1, 28, 7), (1, 29, 1), (1, 29, 4), (1, 29, 6), (1, 30, 3), (1, 30, 5), (1, 31, 4), (1, 31, 8), (1, 32, 3), (1, 32, 6), (1, 33, 1), (1, 34, 0), (1, 34, 1), (1, 34, 4), (1, 35, 6), (1, 36, 0), (1, 36, 1), (1, 37, 0), (1, 37, 4), (1, 37, 7), (1, 38, 0), (1, 38, 6), (1, 39, 6), (1, 39, 7), (1, 39, 8), (1, 40, 0), (1, 40, 5), (1, 40, 6), (1, 41, 8), (1, 42, 1), (1, 42, 2), (1, 42, 4), (1, 42, 5), (1, 42, 7), (1, 44, 6), (1, 44, 7), (1, 45, 1), (1, 45, 5), (1, 46, 0), (1, 46, 4), (1, 47, 8)]

act = Binarize01Act
liste_fonctions.append(act(T=args.thr_bin_act_test[0]))
preprocessing = nn.Sequential(*liste_fonctions).eval()
preprocessing_withoutact = nn.Sequential(*liste_fonctions2).eval()
preprocessing_withoutact_P = nn.Sequential(*liste_fonctions_P).eval()
preprocessing_withoutact_N = nn.Sequential(*liste_fonctions_N).eval()
print(preprocessing)

# Last layer
Wbin_scale = 1.0 * (np.loadtxt(args.path_load_model + "/Wbin_scale.txt").astype("f"))
W_LR = 1.0 * (np.loadtxt(args.path_load_model + "/Wbin.txt").astype("f"))
scale_WLR = 1.0 * (np.loadtxt(args.path_load_model + "/gamma_Wbin.txt").astype("f"))
b_LR = 1.0 * (np.loadtxt(args.path_load_model + "/biais.txt").astype("f"))
# ok

# Unfold and Mapping
unfold_all = {}
for numblockici in range(len(args.type_blocks)):
    unfold_all[numblockici] = [
            torch.nn.Unfold(kernel_size=args.kernel_size_per_block[numblockici], stride=args.Blocks_strides[numblockici],
                            padding=args.padding_per_block[numblockici])]

mapping_filter, input_dim = get_mapping_filter(args)
print(mapping_filter)
if config_general.dataset=="CIFAR10":
    for i in range(16):
        mapping_filter[0][i]=0
    for i in range(16):
        mapping_filter[0][i+16]=1
    for i in range(16):
        mapping_filter[0][i+32]=2

print(mapping_filter)
putawayliteral = []

all_TT_vold, all_TT_noiseonly, nogolist = load_TT_TTnoise(args)
print("START LOADING TT")
array_block_0, array_block_1, nogolist = load_cnf_dnf_block(args)
print(nogolist)
total = 0
soublock = 1
correct = 0
all_TT_noiseonly_block0 = []
for i in range(args.Blocks_filters_output[0]):
    noise_b_f = all_TT_noiseonly[0][i][:,0]
    if len(noise_b_f) == 1:
        noise_b_f = np.array([noise_b_f[0]]*len(all_TT_noiseonly_block0[-1]))
    elif (0,i) in nogolist:
        noise_b_f = np.array([noise_b_f[0]] * len(all_TT_noiseonly_block0[-1]))
    else:
        all_TT_noiseonly_block0.append(noise_b_f)
all_TT_noiseonly_block0 = np.array(all_TT_noiseonly_block0)
all_TT_noiseonly_block1 = []
for i in range(args.Blocks_filters_output[1]):
    noise_b_f = all_TT_noiseonly[1][i][:,0]
    #print(i, len(noise_b_f))
    if len(noise_b_f) == 1:
        noise_b_f = np.array([noise_b_f[0]]*len(all_TT_noiseonly_block0[-1]))
    elif (1,i) in nogolist:
        noise_b_f = np.array([noise_b_f[0]] * len(all_TT_noiseonly_block0[-1]))
    else:
        all_TT_noiseonly_block1.append(noise_b_f)
all_TT_noiseonly_block1 = np.array(all_TT_noiseonly_block1)

print(all_TT_noiseonly_block0.shape)
print(all_TT_noiseonly_block1.shape)
path_save_modelvf = args.path_load_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                            "") + '/sans_contradiction/'
items = [filter_no for filter_no in range(args.Blocks_filters_output[1])]

if config_general.dataset=="MNIST":
    inputs = torch.zeros((1,1,28,28))
else:
    inputs = torch.zeros((1,3,32,32))


predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
shape_all_tensorinput_block, shape_all_tensoroutput_block, res_all_tensorinput_block_unfold = infer_normal_withPYTHON(
    inputs, preprocessing, device, unfold_all, args, mapping_filter, W_LR, b_LR, array_block_0, array_block_1,
    items, putawayliteral)

block_ref_all_inputs, block_ref_all_outputs, cptfinal = get_refs_all2(shape_all_tensorinput_block,
                                                                      shape_all_tensoroutput_block)


dictionnary_ref = get_dictionnary_ref(args, block_ref_all_inputs, block_ref_all_outputs, unfold_all,
                                      mapping_filter)
features1_ref = block_ref_all_outputs[list(block_ref_all_outputs.keys())[-1]].reshape(-1).clone().numpy().astype('i')

H_b0 = deepcopy(block_ref_all_outputs[0]).unsqueeze(0).shape[-1]
H_b1 = deepcopy(block_ref_all_outputs[1]).unsqueeze(0).shape[-1]

unfold_block = unfold_all[0][0]
block_ref_all_inputs_unfold = unfold_block(block_ref_all_inputs[0].unsqueeze(0)).cpu().numpy().astype("i").transpose(1,
                                                                                                                     0,
                                                                                                                     2).reshape(
    9, -1)
block_ref_all_inputs_unfold_b1 = {}
for filter_occurence in tqdm(range(args.Blocks_filters_output[1])):
    if (1, filter_occurence) not in nogolist:
        input_vu_par_cnn_avant_unfold = block_ref_all_inputs[1].unsqueeze(0)[:,
                                        mapping_filter[1][filter_occurence]
                                        : mapping_filter[1][filter_occurence] + int(
                                            args.groups_per_block[1]),
                                        :,
                                        :]
        input_vu_par_cnn_et_sat_starting_newArray = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype(
            "i")
        block_ref_all_inputs_unfold_b1[filter_occurence] = input_vu_par_cnn_et_sat_starting_newArray.transpose(1, 0,
                                                                                                               2).reshape(
            9, -1)

from pysat.solvers import Lingeling, Glucose3, Glucose4, Minisat22, Cadical, MapleChrono, MapleCM, Maplesat, Solver, \
    Minicard, MinisatGH

if args.sat_solver == "Minicard":
    all_solvers = [[Minicard() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "Glucose3":
    all_solvers = [[Glucose3() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "Glucose4":
    all_solvers = [[Glucose4() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "Minisat22":
    all_solvers = [[Minisat22() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "Lingeling":
    all_solvers = [[Lingeling() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "CaDiCaL":
    all_solvers = [[Cadical() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "MapleChrono":
    all_solvers = [[MapleChrono() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "MapleCM":
    all_solvers = [[MapleCM() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "Maplesat":
    all_solvers = [[Maplesat() for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "Mergesat3":
    all_solvers = [[Solver("mergesat3") for j in range(10)] for i in range(args.batch_size_test)]
elif args.sat_solver == "MinisatGH":
    all_solvers = [[MinisatGH() for j in range(10)] for i in range(args.batch_size_test)]
else:
    raise "PB"

from pysat.pb import PBEnc
from pysat.card import *

# transfo
litsici2 = features1_ref.tolist()
all_lits = {}
all_W = {}
all_Vfinal = {}
dico_clause = {}
CNFcondf = {}
for labelref in range(W_LR.shape[0]):
    all_lits[labelref] = {}
    all_W[labelref] = {}
    all_Vfinal[labelref] = {}
    dico_clause[labelref] = {}
    CNFcondf[labelref] = None
    pool = IDPool(start_from=int(max(litsici2)) + 1)
    for aconcurant in range(W_LR.shape[0]):
        if aconcurant != labelref:
            Wf_l = 1.0 * W_LR[labelref, :]
            Wf_a = 1.0 * W_LR[aconcurant, :]
            Wf_diff = Wf_l - Wf_a
            Wf_diff2 = [int(x) for x in Wf_diff]
            bint = float((b_LR[aconcurant] - b_LR[labelref])) / scale_WLR
            Vfinal = int(np.floor(bint))
            if Vfinal >= 0:
                Vfinal = Vfinal
            else:
                Vfinal = Vfinal
            litsici3 = []
            weightsici3 = []
            for index_litteral, litteral in enumerate(litsici2):
                if Wf_diff[index_litteral] != 0:# and features_replace[batchici, index_litteral] == 2:
                    litsici3.append(litteral)
                    weightsici3.append(Wf_diff2[index_litteral])
            if len(litsici3)>0:

                all_lits[labelref][aconcurant] = litsici3
                all_W[labelref][aconcurant] = weightsici3
                all_Vfinal[labelref][aconcurant] = (int(Vfinal),bint)

                cnflr = PBEnc.leq(lits=litsici3, weights=weightsici3, bound=int(Vfinal),
                              encoding=4, vpool=pool).clauses
            else:
                cnflr = None
            dico_clause[labelref][aconcurant] = deepcopy(cnflr)
            del cnflr
            #del pool
from pysat.formula import IDPool
CNFcondf = {}
litsici2 = features1_ref.tolist()
import pandas as pd

print("START load csv")
df0 = []
for filter_occurencefunction in tqdm(range(args.Blocks_filters_output[0])):
    if (0, filter_occurencefunction) not in nogolist:
        dfinter = pd.read_csv(path_save_modelvf +"/TTnet_allposible_block0_filter"+str(filter_occurencefunction)+".csv", header=None, sep='\n').to_numpy().transpose()[0]
        print(dfinter.shape)
        for ix, x in enumerate(dfinter):
            dfinter[ix] = str(x).replace(" ", "").replace("\n", "").replace(
            '],"', ']","').replace('"', "")
        df0.append(dfinter)
df0 = np.array(df0)
print(df0.shape)
print("END load csv")

print("START load csv")
df1 = []
for filter_occurencefunction in tqdm(range(args.Blocks_filters_output[1])):
    if (1, filter_occurencefunction) not in nogolist:
        dfinter = pd.read_csv(path_save_modelvf +"/TTnet_allposible_block1_filter"+str(filter_occurencefunction)+".csv", header=None, sep='\n').to_numpy().transpose()[0]
        print(dfinter.shape)
        for ix, x in enumerate(dfinter):
            dfinter[ix] = str(x).replace(" ", "").replace("\n", "").replace(
                '],"', ']","').replace('"', "")
        df1.append(dfinter)
df1 = np.array(df1)
print(df1.shape)
print("END load csv")
#
# #print(df[807058])
# #print(df[807059])

import json
import itertools
def extract(output_strings):
    template = '{{"values": [{}]}}'
    parse_func = lambda x: json.loads(template.format(x))
    return list(itertools.chain(*[parse_func(x)["values"] for x in output_strings]))


# dask_df_0 = dd.read_csv(path_save_modelvf + "/TTnet_allposible_block0.csv")
tottime = 0
running_corrects = 0.0
nbre_sample = 0
timeout = 0
tk0 = tqdm(dataloaders["val"], total=int(len(dataloaders["val"])))

def BitsToIntAFast(bits):
    m,n = bits.shape # number of columns is needed, not bits.size
    a = 2**np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code

def TerToIntAFast(bits):
    m,n = bits.shape # number of columns is needed, not bits.size
    #print(m,n)
    a = 3**np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code


correct_sur = 0
preprocessing.eval()
preprocessing_inpQ.eval()
preprocessing_withoutact.eval()
preprocessing_withoutact_P.eval()
preprocessing_withoutact_N.eval()
with torch.no_grad():
    for indexicicici, data in enumerate(tk0):
        if int(args.coef_multiplicateur_data) * (int(args.offset) + 1) > indexicicici >= int(
                args.coef_multiplicateur_data) * int(args.offset):
            nSize = args.kernel_size_per_block[0] ** 2 * args.groups_per_block[0]
            inputs, labels = data
            #start = time.time()

            predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
            shape_all_tensorinput_block, shape_all_tensoroutput_block, res_all_tensorinput_block_unfold = infer_normal_withPYTHON(inputs, preprocessing,
                                                                                                device, unfold_all, args,
                                                                                                mapping_filter, Wbin_scale, b_LR,
                                                                                                array_block_0,
                                                                                                array_block_1, items, putawayliteral)

            batch_size_test = labels.shape[0]
            predicted = torch.Tensor(predicted).to(device)
            imagev2 = inputs[(predicted == labels.to(device)), :, :, :]
            labelrefv2 = labels[(predicted == labels.to(device))].clone().detach().cpu().numpy().astype("i")
            batch_size_test = labelrefv2.shape[0]
            batchtot = {}
            cpt_batchtrue = -1
            for batchici in tqdm(range(predicted.shape[0])):
                if (predicted == labels.to(device))[batchici]:
                    cpt_batchtrue+=1
                    batchtot[cpt_batchtrue] = batchici
            Batch_present_all_v0 = {i: [] for i in range(H_b0 * H_b0 * 3 ** nSize)} #(-1 * np.ones((H_b0 * H_b0 * 3 ** 9, batch_size_test))).astype("i")
            Batch_present_all_v0_ref = {j: {i: [] for i in range(H_b0 * H_b0 * 3 ** nSize)} for j in range(args.Blocks_filters_output[0])}#.astype("i")
            Batch_present_all_v1_ref = {j: {i: [] for i in range(H_b1 * H_b1 * 3 ** nSize)} for j in range(args.Blocks_filters_output[1])}#.astype("i")
            enumerateici_B = [i // (H_b1 * H_b1) for i in range(H_b1 * H_b1 * batch_size_test)]
            enumerateiciB0 = [i // (H_b0 * H_b0) for i in range(H_b0 * H_b0 * batch_size_test)]
            res_all_tensorinput_block_unfold_U = {}
            correct += labelrefv2.shape[0]
            correct_sur += labelrefv2.shape[0]
            total += labelrefv2.shape[0]
            cnf_general = [[] for i in range(batch_size_test)]
            cnf_general_pre = [[] for i in range(batch_size_test)]
            nbre_sample += inputs.shape[0]
            start2 = time.time()
            print(correct)
            imagesPLUS = imagev2.clone() + float(args.attack_eps)
            imagesMOINS = imagev2.clone() - float(args.attack_eps)
            outp = preprocessing(imagesPLUS).to(device).detach().cpu().clone()
            outm = preprocessing(imagesMOINS).to(device).detach().cpu().clone()
            input_acomplter_v2 = (outm * (outm == outp).float() + -1 * (outm != outp))
            print(torch.sum(input_acomplter_v2 == -1) / (input_acomplter_v2.shape[0]))
            print()

            for block_occurence in range(len(args.Blocks_filters_output)): ##Each layer
                iterici = 0
                filtericitot = args.Blocks_filters_output[block_occurence]
                img_fin = None
                unfold_block = unfold_all[block_occurence][iterici]
                kici = args.kernel_size_per_block[block_occurence] * args.kernel_size_per_block[block_occurence] * \
                       args.groups_per_block[block_occurence]
                dicorefblock = dictionnary_ref[block_occurence]
                #print(dicorefblock)
                if block_occurence == 0:
                    imgs_debut = copy(input_acomplter_v2)
                else:
                    imgs_debut = copy(
                        res_all_tensorinput_block[block_occurence])  # [(predicted == labelref.to(device)), :, :, :])
                imgs_fin = res_all_tensoroutput_block[block_occurence][(predicted.cpu() == labels.cpu()), :, :, :]
                shapeici_out = imgs_fin.shape[-1] ** 2
                shapeici_out2 = imgs_fin.shape[-1]
                unfold_block = unfold_all[block_occurence][iterici]
                cpt_bloc0 = -1
                cpt_bloc1 = -1



                if block_occurence == 0:
                    if config_general.dataset=="MNIST":
                        input_vu_par_cnn_avant_unfold = imgs_debut[:,  # batch
                                                        mapping_filter[block_occurence][0]  # channel
                                                        : mapping_filter[block_occurence][0] + int(
                                                            args.groups_per_block[block_occurence]),  # channel
                                                        :,
                                                        :]
                        input_vu_par_cnn_et_sat_starting = unfold_block(
                            input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                        input_vu_par_cnn_et_sat_starting_newArray = deepcopy(input_vu_par_cnn_et_sat_starting)
                        input_vu_par_cnn_et_sat_starting_newArray[input_vu_par_cnn_et_sat_starting == -1] = 2
                        input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting_newArray.transpose(1, 0, 2).reshape(
                            nSize, -1)
                        #for filter_occurenceb0 in range(args.Blocks_filters_output[0]):
                        #    for kkb0 in range(9):
                        #        if (block_occurence, filter_occurenceb0, kkb0) in putawayliteral:
                        #            input_vu_par_cnn_et_sat[kkb0, :] = 0
                        #            print(np.sum(input_vu_par_cnn_et_sat)==2)
                        input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                        input_vu_par_cnn_et_sat_T_int = TerToIntAFast(input_vu_par_cnn_et_sat_T)
                        for input_vu_par_cnn_et_sat_T_int_i, input_vu_par_cnn_et_sat_T_int_v in enumerate(
                                input_vu_par_cnn_et_sat_T_int):
                            input_vu_par_cnn_et_sat_T_int[
                                input_vu_par_cnn_et_sat_T_int_i] = input_vu_par_cnn_et_sat_T_int_v + (
                                        input_vu_par_cnn_et_sat_T_int_i % (H_b0 * H_b0)) * (3 ** nSize)
                            Batch_present_all_v0[input_vu_par_cnn_et_sat_T_int[input_vu_par_cnn_et_sat_T_int_i]].append(
                                enumerateiciB0[input_vu_par_cnn_et_sat_T_int_i])
                        value_for_TTnoise_np_reshape_unique, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                                         return_inverse=True)  # ,axis=1)
                        recomposition_val = [x % (3 ** nSize) for ix, x in enumerate(value_for_TTnoise_np_reshape_unique)]
                        cnf_noise_only = all_TT_noiseonly_block0[:, recomposition_val]
                        output_filters = np.zeros((args.Blocks_filters_output[0], len(recomposition_val)))
                        output_filters[cnf_noise_only == b'U'] = 2
                        output_var_unfold = output_filters[:, inverse_indices]
                        img_fin_unknown = torch.Tensor(
                            output_var_unfold.reshape(args.Blocks_filters_output[0], batch_size_test, H_b0, H_b0).transpose(1, 0, 2, 3)).to(device)

                        all_clause_ici = df0[:,value_for_TTnoise_np_reshape_unique]
                        #print(all_clause_ici.shape)
                        cpt_filter_icib0 = -1
                        for filter_occurenceb0 in range(args.Blocks_filters_output[0]):
                            if (0, filter_occurenceb0) not in nogolist:
                                cpt_filter_icib0+=1
                                all_clause_ici_2 = all_clause_ici[cpt_filter_icib0]
                                #print(all_clause_ici_2.shape)
                                for xyval_pixel_i, xyval_pixel_v in enumerate(value_for_TTnoise_np_reshape_unique):
                                    cnf_ici = all_clause_ici_2[xyval_pixel_i]
                                    if cnf_ici != '-':
                                        Batch_present = Batch_present_all_v0[xyval_pixel_v]  # Batch_present_all[xyval_pixel]
                                        for batchici2 in Batch_present:
                                            cnf_general[batchici2].append(cnf_ici)

                    elif config_general.dataset=="CIFAR10":
                        res_all_tensorinput_block_unfold_U[0] = {}
                        doit = 48 #int(args.preprocessing_CNN[0] / args.groups_per_block[block_occurence])
                        coef_multi = 1 #int(args.Blocks_filters_output[0] / doit)
                        img_fin_unknown = []
                        for filter_occurenceb0 in range(args.Blocks_filters_output[0]):
                            if (0, filter_occurenceb0) not in nogolist:

                                cpt_bloc0+=1
                                Batch_present_all_v0 = Batch_present_all_v0_ref[filter_occurenceb0]
                                input_vu_par_cnn_avant_unfold = imgs_debut[:,  # batch
                                                                mapping_filter[block_occurence][
                                                                    coef_multi * filter_occurenceb0]  # channel
                                                                : mapping_filter[block_occurence][
                                                                      coef_multi * filter_occurenceb0] + int(
                                                                    args.groups_per_block[block_occurence]),  # channel
                                                                :,
                                                                :]
                                input_vu_par_cnn_et_sat_starting_newArray = unfold_block(
                                    input_vu_par_cnn_avant_unfold).cpu().numpy().astype(
                                    "i")
                                input_vu_par_cnn_et_sat_starting_newArray[input_vu_par_cnn_et_sat_starting_newArray == -1] = 2
                                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting_newArray.transpose(1, 0, 2).reshape(
                                    nSize, -1)
                                input_vu_par_cnn_et_sat_U = input_vu_par_cnn_et_sat_starting_newArray.transpose(1, 0, 2)
                                for kkb0 in range(9):
                                    if (0, filter_occurenceb0, kkb0) in putawayliteral:
                                        input_vu_par_cnn_et_sat[kkb0, :] = 0
                                        input_vu_par_cnn_et_sat_U[kkb0, :, :] = 0
                                        #print(np.sum(input_vu_par_cnn_et_sat==2))
                                res_all_tensorinput_block_unfold_U[0][filter_occurenceb0] = input_vu_par_cnn_et_sat_U
                                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                                input_vu_par_cnn_et_sat_T_int = TerToIntAFast(input_vu_par_cnn_et_sat_T)
                                for input_vu_par_cnn_et_sat_T_int_i, input_vu_par_cnn_et_sat_T_int_v in enumerate(
                                        input_vu_par_cnn_et_sat_T_int):
                                    input_vu_par_cnn_et_sat_T_int[
                                        input_vu_par_cnn_et_sat_T_int_i] = input_vu_par_cnn_et_sat_T_int_v + (
                                            input_vu_par_cnn_et_sat_T_int_i % (H_b0 * H_b0)) * (3 ** nSize)
                                    Batch_present_all_v0[input_vu_par_cnn_et_sat_T_int[input_vu_par_cnn_et_sat_T_int_i]].append(
                                        enumerateiciB0[input_vu_par_cnn_et_sat_T_int_i])
                                value_for_TTnoise_np_reshape_unique, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                                                 return_inverse=True)  # ,axis=1)
                                recomposition_val = [x % (3 ** nSize) for ix, x in enumerate(value_for_TTnoise_np_reshape_unique)]
                                cnf_noise_only = all_TT_noiseonly_block0[cpt_bloc0, recomposition_val]
                                output_filters = np.zeros(len(recomposition_val))
                                output_filters[cnf_noise_only == b'U'] = 2
                                output_var_unfold = output_filters[inverse_indices]
                                #print(np.sum(output_var_unfold==2))
                                img_fin_unknown.append(output_var_unfold.reshape(1, batch_size_test, H_b0, H_b0))
                                #print(value_for_TTnoise_np_reshape_unique, df0[filter_occurenceb0].shape)
                                all_clause_ici = df0[cpt_bloc0][value_for_TTnoise_np_reshape_unique]
                                for xyval_pixel_i, xyval_pixel_v in enumerate(value_for_TTnoise_np_reshape_unique):
                                    cnf_ici = all_clause_ici[xyval_pixel_i]
                                    if cnf_ici != '-':
                                        Batch_present = Batch_present_all_v0[xyval_pixel_v]  # Batch_present_all[xyval_pixel]
                                        #print(filter_occurenceb0, len(Batch_present), len(cnf_general), Batch_present)
                                        for batchici2 in Batch_present:
                                            #print(batchici2)
                                            cnf_general[batchici2].append(cnf_ici)

                            else:
                                img_fin_unknown.append(np.zeros((1, batch_size_test, H_b0, H_b0)))
                        #print(cnf_general[0])
                        img_fin_unknown = np.vstack(img_fin_unknown).transpose(1, 0, 2, 3)



                    imgs_fin = imgs_fin+img_fin_unknown
                    imgs_fin[imgs_fin > 1] = 2
                    res_all_tensorinput_block[block_occurence+1] = deepcopy(imgs_fin)
                    del imgs_debut
                    imgs_debut = deepcopy(imgs_fin)
                    startlocal = time.time()
                    print("Block occurence, mean noise", block_occurence, torch.sum(imgs_fin == 2) / batch_size_test)
                    print("from start verify 3", time.time() - start2)

                elif block_occurence == 1:
                    img_fin_unknown = []
                    res_all_tensorinput_block_unfold_U[1] = {}
                    for filter_occurence in tqdm(range(args.Blocks_filters_output[1])):
                        if (1, filter_occurence) not in nogolist:
                            cpt_bloc1 += 1
                            Batch_present_all_v1 = Batch_present_all_v1_ref[filter_occurence]
                            input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                            mapping_filter[1][filter_occurence]
                                                            : mapping_filter[1][filter_occurence] + int(
                                                                args.groups_per_block[1]),
                                                            :,
                                                            :]
                            input_vu_par_cnn_et_sat_starting_newArray = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype(
                                "i")
                            input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting_newArray.transpose(1, 0, 2).reshape(nSize,-1)
                            input_vu_par_cnn_et_sat_U = input_vu_par_cnn_et_sat_starting_newArray.transpose(1, 0, 2)
                            for kkb1 in range(9):
                                if (1, filter_occurence, kkb1) in putawayliteral:
                                    input_vu_par_cnn_et_sat[kkb1, :] = 0
                                    input_vu_par_cnn_et_sat_U[kkb1, :, :] = 0

                            res_all_tensorinput_block_unfold_U[1][filter_occurence] = input_vu_par_cnn_et_sat_U

                            input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                            input_vu_par_cnn_et_sat_T_int = TerToIntAFast(input_vu_par_cnn_et_sat_T)




                            for input_vu_par_cnn_et_sat_T_int_i, input_vu_par_cnn_et_sat_T_int_v in enumerate(
                                    input_vu_par_cnn_et_sat_T_int):
                                input_vu_par_cnn_et_sat_T_int[
                                    input_vu_par_cnn_et_sat_T_int_i] = input_vu_par_cnn_et_sat_T_int_v + (
                                            input_vu_par_cnn_et_sat_T_int_i % (H_b1 * H_b1)) * (3 ** nSize)
                                Batch_present_all_v1[input_vu_par_cnn_et_sat_T_int[input_vu_par_cnn_et_sat_T_int_i]].append(enumerateici_B[input_vu_par_cnn_et_sat_T_int_i])
                            value_for_TTnoise_np_reshape_unique, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                                             return_inverse=True)  # ,axis=1)
                            recomposition_val = [x % (3 ** nSize) for ix, x in enumerate(value_for_TTnoise_np_reshape_unique)]
                            cnf_noise_only = all_TT_noiseonly_block1[cpt_bloc1, recomposition_val]
                            output_filters = np.zeros(len(recomposition_val))
                            output_filters[cnf_noise_only == b'U'] = 2
                            output_var_unfold = output_filters[ inverse_indices]
                            img_fin_unknown.append(output_var_unfold.reshape(1, batch_size_test, H_b1, H_b1))
                            all_clause_ici = df1[cpt_bloc1][value_for_TTnoise_np_reshape_unique]
                            for xyval_pixel_i, xyval_pixel_v in enumerate(value_for_TTnoise_np_reshape_unique):
                                cnf_ici = all_clause_ici[xyval_pixel_i]
                                if cnf_ici != '-':
                                    Batch_present = Batch_present_all_v1[xyval_pixel_v]  # Batch_present_all[xyval_pixel]
                                    for batchici2 in Batch_present:
                                        cnf_general[batchici2].append(cnf_ici)
                        else:
                            img_fin_unknown.append(np.zeros((1, batch_size_test, H_b1, H_b1)))
                    img_fin_unknown = np.vstack(img_fin_unknown).transpose(1, 0, 2, 3)
                    imgs_fin = imgs_fin + img_fin_unknown
                    imgs_fin[imgs_fin > 1] = 2
                    print("Block occurence, mean noise", block_occurence, np.sum(imgs_fin == 2) / batch_size_test)
                    #print(cnf_general[0])
                    #print("from start", time.time() - start)
                    print("from start verify 3", time.time() - start2)
                    cnf_general2 = [extract(cnf) for cnf in cnf_general]
                    #print("from start", time.time() - start)
                    print("from start verify 3", time.time() - start2)
                    #[all_solvers[index_cnf][j].append_formula(cnf) for j in range(nSize) for index_cnf, cnf in enumerate(cnf_general2)]
                    #print(all_solvers[0])
                    #print(all_solvers[0][0].nof_clauses())
                    #print(len(cnf_general2[0]))
                    #print("from start", time.time() - start)
                    #print("from start verify 3", time.time() - start2)
                    #print("Block occurence, mean noise", block_occurence, np.sum(imgs_fin == 2) / batch_size_test)
                    #print("Block occurence, mean noise", block_occurence, np.sum(imgs_fin == 2) / (batch_size_test * args.Blocks_filters_output[block_occurence] * H_b1 * H_b1))
                    features_replace = imgs_fin.reshape(batch_size_test, -1).astype('i')
                    KNOWN_all = (features_replace < 2)
                    UNKNOWN_all = (features_replace == 2)
                    KNOWN_all_0 = (features_replace == 0)
                    KNOWN_all_1 = (features_replace == 1)
                    timesatsolve_all = []
                    time_to_remove_all = 0
                    time_to_solve_all = 0
                    #imgs_fin_offset = deepcopy(imgs_fin)
                    #imgs_fin_offset[imgs_fin == 2] = 0
                    #features_replace_offset = imgs_fin_offset.reshape(batch_size_test, -1).astype(
                    #    'i').transpose()
                    #offset_F = np.dot(W_LR, features_replace_offset).transpose()
                    #print(offset_F.shape)

                    # print("START ASSERT 1")
                    # for batchici in tqdm(range(features_replace.shape[0])):
                    #     cnf_local = []
                    #     for index_x, x in enumerate(features_replace[batchici, :]):
                    #         if x == 1:
                    #             cnf_local.append([litsici2[index_x]])
                    #         else:
                    #             cnf_local.append([int(-1 * litsici2[index_x])])
                    #     labelref_vf = None
                    #     for labelref in range(W_LR.shape[0]):
                    #         if CNFcondf[labelref] is not None:
                    #             cnf = CNFcondf[labelref] + cnf_local
                    #             with Solver(bootstrap_with=cnf) as solver:
                    #                 if solver.solve():
                    #                     labelref_vf = labelref
                    #                     #break
                    #                     assert labelref_vf == labelrefv2[batchici] #$or V_ref[0][labelref] == V_ref[0][predicted[0]]
                    #                     break
                    #             del cnf
                    #     del cnf_local




                    # for batchici in tqdm(range(batch_size_test)):
                    #     UNKNOWN1 = UNKNOWN_all[batchici]
                    #     KNOWN_all_0_ici = KNOWN_all_0[batchici]
                    #     KNOWN_all_1_ici = KNOWN_all_1[batchici]
                    #     cnf_G = cnf_general2[batchici] #+ cnf_general_pre[batchici]
                    #     cnf_pre = cnf_general_pre[batchici]
                    #     cnf_local = []
                    #
                    #     offset = features1_ref[0]
                    #     if np.sum(UNKNOWN1) != 0:
                    #         for k0ici_index, k0ici in enumerate(KNOWN_all_0_ici):
                    #             if k0ici:
                    #                 cnf_local.append([int(-1*features1_ref[k0ici_index])])
                    #                 #assert True not in [features1_ref[k0ici_index] in x for x in cnf_G]
                    #                 #assert True not in [int(-1*features1_ref[k0ici_index]) in x for x in cnf_G]
                    #             elif KNOWN_all_1_ici[k0ici_index]:
                    #                 cnf_local.append([int(1*features1_ref[k0ici_index])])
                    #                 #assert True not in [features1_ref[k0ici_index] in x for x in cnf_G]
                    #                 #assert True not in [int(-1 * features1_ref[k0ici_index]) in x for x in cnf_G]
                    #             #else:
                    #                 #assert UNKNOWN1[k0ici_index] == True
                    #         L0 = np.expand_dims(-1 * (np.where(features_replace[batchici] == 0)[0] + offset),
                    #                        axis=1).tolist()
                    #         L1 = np.expand_dims(
                    #             1 * (np.where(features_replace[batchici] == 1)[0] + offset), axis=1).tolist()
                    #         #print(L0 + L1)
                    #         #print(cnf_local)
                    #         #ok

                    #print("from start verify 3", time.time() - start2)
                    #ok




                    cnf_local_all = {}
                    offset = features1_ref[0]
                    for batchici in tqdm(range(batch_size_test)):
                        cnf_local_all[batchici] = np.expand_dims(-1 * (np.where(features_replace[batchici] == 0)[0] + offset), axis=1).tolist()
                        cnf_local_all[batchici] += np.expand_dims(1 * (np.where(features_replace[batchici] == 1)[0] + offset), axis=1).tolist()
                    print("from start verify 3", time.time() - start2)
                    #ok

                    for batchici in tqdm(range(batch_size_test)):
                        UNKNOWN1 = UNKNOWN_all[batchici]
                        if np.sum(UNKNOWN1) != 0:
                            cnf_local = cnf_local_all[batchici]
                            cnf_G = cnf_general2[batchici]  # + cnf_general_pre[batchici]
                            cnf_pre = cnf_general_pre[batchici]
                            flag2, time_to_remove, time_to_solve, attack, new_lab = complete_solving_accelerate_v3(
                                                                        dico_clause[labelrefv2[batchici]], labelrefv2[batchici],
                                                                        all_solvers[batchici], cnf_G + cnf_local, cnf_pre, path_save_modelvf_str=path_save_modelvf+"cnf/"+str(args.attack_eps)+"_Batch_correct_"+str(batchtot[batchici])+"_")
                            #del cnf_G, cnf_local, cnf_pre
                            time_to_remove_all+=time_to_remove
                            time_to_solve_all += time_to_solve
                            #print(ok)
                            if flag2:
                                start_assert = time.time()
                                all_CNFG[labelrefv2[batchici]][new_lab] += cnf_G
                                #print(cnf_G)
                                #print()
                                #print(attack)
                                #print()
                                #print([x for x in cnf_G if (804 in x or -804 in x)])
                                #print()
                                inputs_attack = imagev2.clone()
                                correct -= 1
                                cpt_attack = 0
                                for c in range(1):
                                    for i in range(28):
                                        for j in range(28):
                                            cpt_attack+=1
                                            if cpt_attack in attack: #and inputs_attack[batchici, c, i, j]==-1:
                                                inputs_attack[batchici, c, i, j]  = inputs_attack[batchici, c, i, j] + args.attack_eps
                                            elif int(-1*cpt_attack): # in attack and inputs_attack[batchici, c, i, j]==-1:
                                                inputs_attack[batchici, c, i, j] = inputs_attack[batchici, c, i, j] - args.attack_eps

                                            #if cpt_attack == 123:
                                            #    print(imagev2[batchici])
                                            #    print(inputs_attack[batchici])



                                predicted_attack, res_all_tensorinput_block_attack, res_all_tensoroutput_block_attack, _, _,\
                                        res_all_tensorinput_block_unfold_attack = infer_normal_withPYTHON(inputs_attack[batchici].unsqueeze(0),
                                                               preprocessing,
                                                               device, unfold_all, args,
                                                               mapping_filter, Wbin_scale, b_LR,
                                                               array_block_0,
                                                               array_block_1, items, putawayliteral)

                                predicted_ref, res_all_tensorinput_block_ref, res_all_tensoroutput_block_ref, _, _,\
                                        res_all_tensorinput_block_unfold_ref = infer_normal_withPYTHON(imagev2[batchici].unsqueeze(0),
                                                               preprocessing,
                                                               device, unfold_all, args,
                                                               mapping_filter, Wbin_scale, b_LR,
                                                               array_block_0,
                                                               array_block_1, items, putawayliteral)

                                #print(predicted_attack[0],  labelrefv2[batchici], new_lab, predicted_ref[0])
                                if predicted_attack[0] != labelrefv2[batchici]:
                                    correct_sur -= 1
                                        #TODOO : save les images avec et sans attaque apres la premiere couche
                                else:
                                    print("correct", correct)
                                    print("correct_sur", correct_sur)

                                    features_replace_attack = res_all_tensoroutput_block_attack[1].reshape(1,
                                                                                                           -1).astype(
                                        'i').transpose()
                                    features_replace_ref = res_all_tensoroutput_block_ref[1].reshape(1, -1).astype(
                                        'i').transpose()
                                    V_ref_aattack = np.dot(Wbin_scale, features_replace_attack).transpose() + b_LR
                                    V_ref_ref = np.dot(Wbin_scale, features_replace_ref).transpose() + b_LR

                                time_to_remove_all += time.time() - start_assert

                    print("natural correct", correct)
                    print("verifed correct", correct)
                    print("verifed correct with verification", correct_sur)
                    print("from start", time.time() - start2)
                    print("TIME TO GENERATE Verify ", time.time() - start2-time_to_remove_all)
                    print("TIME TO SOLVE Verify ", time_to_solve_all)
                    # print(all_CNFG)

                    print(" DONE ")
                    #print(ok)














