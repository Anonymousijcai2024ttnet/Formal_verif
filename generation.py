import os
from copy import copy, deepcopy
import time

from src.inference import complete_solving_accelerate_v3, BN_eval_MNIST, BN_eval_CIFAR10, load_cnf_dnf_block, \
    TerToIntAFast, extract

from src.inference import get_mapping_filter, Binarize01Act, InputQuantizer, load_data, load_cnf_dnf, \
    infer_normal_withPYTHON, get_refs_all2, BN_eval_CNN, get_mapping_filter_cnn, get_dictionnary_ref, transform_cnf, \
    get_refs_all_cnn
import torch
import argparse
from ctypes import *
from tqdm import tqdm
import concurrent.futures
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







#Preprocessing
liste_fonctions = []
liste_fonctions.append(InputQuantizer(args.quant_step))
if args.first_layer=="BNs":
    scale = np.loadtxt(args.path_load_model+"/preprocessing_BN_scale.txt")
    bias = np.loadtxt(args.path_load_model+"/preprocessing_BN_bias.txt")
    if config_general.dataset == "MNIST":
        liste_fonctions.append(BN_eval_MNIST(np.array([scale]), np.array([bias])).to(device))
    else:
        liste_fonctions.append(BN_eval_CIFAR10(np.array([scale]), np.array([bias])).to(device))
    # liste_fonctions.append(BN_eval(np.array([scale]),np.array([bias])).to(device))
    if os.path.isfile(args.path_load_model + "/preprocessing_BN_scale_2.txt"):
        scale2 = np.loadtxt(args.path_load_model + "/preprocessing_BN_scale_2.txt")
        bias2 = np.loadtxt(args.path_load_model + "/preprocessing_BN_bias_2.txt")
        if config_general.dataset == "MNIST":
            liste_fonctions.append(BN_eval_MNIST(np.array([scale2]), np.array([bias2])).to(device))
        else:
            liste_fonctions.append(BN_eval_CIFAR10(np.array([scale2]), np.array([bias2])).to(device))

elif args.first_layer == "bin":
    cnn = 1.0 * (np.loadtxt(args.path_load_model + "/cnn.txt").astype("f"))
    #biais2 = 1.0 * (np.loadtxt(args.path_load_model + "/cnn_biais.txt").astype("f"))
    if config_general.dataset == "MNIST":
        cnn = torch.Tensor(cnn).reshape(args.preprocessing_CNN[0], 1, args.preprocessing_CNN[1], args.preprocessing_CNN[1]).to(device)
    else:
        cnn = torch.Tensor(cnn).reshape(args.preprocessing_CNN[0], 3, args.preprocessing_CNN[1], args.preprocessing_CNN[1]).to(
        device)
    print(cnn.shape)
    cnnici = torch.nn.Conv2d(cnn.shape[1], cnn.shape[0], kernel_size = cnn.shape[-1], stride=args.preprocessing_CNN[2], bias=False)
    print(cnn.shape)
    cnnici.weight.data = cnn.clone()
    #cnnici.bias.data = torch.Tensor(biais2).to(device)
    liste_fonctions.append(cnnici)
    scale = np.loadtxt(args.path_load_model+"/preprocessing_BN_scale.txt")
    bias = np.loadtxt(args.path_load_model+"/preprocessing_BN_bias.txt")
    liste_fonctions.append(BN_eval_CNN(np.array([scale]), np.array([bias])).to(device))
    #print(np.array(scale), np.array(bias))


#ok

act = Binarize01Act
liste_fonctions.append(act(T=args.thr_bin_act_test[0]))
preprocessing = nn.Sequential(*liste_fonctions).eval()
print(preprocessing)

#Last layer
Wbin_scale = 1.0*(np.loadtxt(args.path_load_model+"/Wbin_scale.txt").astype("f"))
W_LR = 1.0*(np.loadtxt(args.path_load_model+"/Wbin.txt").astype("f"))
scale_WLR = 1.0*(np.loadtxt(args.path_load_model+"/gamma_Wbin.txt").astype("f"))
print(scale_WLR)
b_LR = 1.0*(np.loadtxt(args.path_load_model+"/biais.txt").astype("f"))
print(W_LR, np.unique(W_LR), b_LR)


# Unfold and Mapping
unfold_all = {}
for numblockici in range(len(args.type_blocks)):
    unfold_all[numblockici] = [
            torch.nn.Unfold(kernel_size=args.kernel_size_per_block[numblockici], stride=args.Blocks_strides[numblockici],
                            padding=args.padding_per_block[numblockici])]
if args.first_layer=="BNs":
    mapping_filter, input_dim = get_mapping_filter(args)
elif args.first_layer == "bin":
    mapping_filter, input_dim = get_mapping_filter_cnn(args)

if config_general.dataset == "CIFAR10":
    for i in range(16):
        mapping_filter[0][i+16] = 1
    for i in range(16):
        mapping_filter[0][i+32] = 2
print(mapping_filter)





all_TT_vold, all_TT_noiseonly, nogolist = load_TT_TTnoise(args)
#print(all_TT_vold)
# nogolist.append((0,27))
# nogolist.append((1,27))
print(nogolist)
total = 0
soublock = 1
correct = 0
all_TT_noiseonly_block0 = []
for i in range(args.Blocks_filters_output[0]):
    noise_b_f = all_TT_noiseonly[0][i][:,0]
    all_TT_noiseonly_block0.append(noise_b_f)
all_TT_noiseonly_block0 = np.array(all_TT_noiseonly_block0)
all_TT_noiseonly_block1 = []
for i in range(args.Blocks_filters_output[1]):
    noise_b_f = all_TT_noiseonly[1][i][:,0]
    all_TT_noiseonly_block1.append(noise_b_f)
all_TT_noiseonly_block1 = np.array(all_TT_noiseonly_block1)





path_save_modelvf = args.path_load_model+'/thr_'+str(args.thr_bin_act_test[1:]).replace(" ","")+'/sans_contradiction/'





print("START LOADING TT")

array_block_0, array_block_1, nogolist = load_cnf_dnf_block(args)
#nogolist.append((0,27))
print(nogolist)

items = [filter_no for filter_no in range(args.Blocks_filters_output[1])]

if config_general.dataset=="MNIST":
    inputs = torch.zeros((1,1,28,28))
else:
    inputs = torch.zeros((1,3,32,32))

predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
shape_all_tensorinput_block, shape_all_tensoroutput_block, _ = infer_normal_withPYTHON(inputs, preprocessing, device, unfold_all, args, mapping_filter, W_LR, b_LR, array_block_0, array_block_1, items)

#print(res_all_tensorinput_block)
#ok
if args.first_layer=="BNs":
    block_ref_all_inputs, block_ref_all_outputs, cptfinal = get_refs_all2(shape_all_tensorinput_block,
                                                                      shape_all_tensoroutput_block)
elif args.first_layer == "bin":
    if config_general.dataset == "MNIST":
        coef_startici = 28*28
    else:
        coef_startici = 3*32*32
    block_ref_all_inputs, block_ref_all_outputs, cptfinal = get_refs_all_cnn(shape_all_tensorinput_block,
                                                                      shape_all_tensoroutput_block, coef_start=coef_startici)

dictionnary_ref = get_dictionnary_ref(args, block_ref_all_inputs, block_ref_all_outputs, unfold_all,
                        mapping_filter)
features1_ref = block_ref_all_outputs[list(block_ref_all_outputs.keys())[-1]].reshape(-1).clone().numpy().astype('i')

H_b0 = deepcopy(block_ref_all_outputs[0]).unsqueeze(0).shape[-1]
H_b1 = deepcopy(block_ref_all_outputs[1]).unsqueeze(0).shape[-1]

print(H_b0, H_b1)
#print(dictionnary_ref)
#print(features1_ref)
#ok


import csv

# with open( path_save_modelvf +"/TTnet_allposible_block0.csv", 'w') as csvfile:
#     nSize = args.kernel_size_per_block[0] ** 2 * args.groups_per_block[0]
#
#     writer = csv.writer(csvfile)
#     cpt_total = 0
#     for xpixel in tqdm(range(H_b0)):
#         for ypixel in range(H_b0):
#             cpt = xpixel*H_b0+ypixel #41
#             for enterbase2i in range(3 ** nSize):
#                 #enterbase2i = 55
#                 TT_filter_v2 = []
#                 for filter_occurencefunction in range(args.Blocks_filters_output[0]):
#                     if (0, filter_occurencefunction) not in nogolist:
#                         all_TT_noiseonlyici = all_TT_noiseonly[0][filter_occurencefunction]
#                         print(all_TT_noiseonlyici.shape, filter_occurencefunction)
#                     if (1, filter_occurencefunction) not in nogolist:
#                         all_TT_noiseonlyicinotuse = all_TT_noiseonly[1][filter_occurencefunction]
#                         print(all_TT_noiseonlyicinotuse.shape, filter_occurencefunction)
#                         if all_TT_noiseonlyici[enterbase2i]== b'U':
#                             all_TTici = all_TT_vold[0][filter_occurencefunction]
#                             dicorefblockfilter = dictionnary_ref[0][filter_occurencefunction]
#                             expressionici = all_TTici[enterbase2i]
#                             input_varref, output_binary_ref = dicorefblockfilter[cpt]
#                             val_cnf_ici = transform_cnf(input_varref, output_binary_ref,
#                                                                  expressionici, [], k=nSize)
#                             TT_filter_v2+=val_cnf_ici
#                 if len(TT_filter_v2)==0:
#                     writer.writerow(["-"])
#                 else:
#                     writer.writerow(TT_filter_v2)



for filter_occurencefunction in tqdm(range(48)): #args.Blocks_filters_output[0])):
    nSize = args.kernel_size_per_block[0] ** 2 * args.groups_per_block[0]
    if (0, filter_occurencefunction) not in nogolist:
        with open(path_save_modelvf +"/TTnet_allposible_block0_filter"+str(filter_occurencefunction)+".csv", 'w') as csvfile:
            writer = csv.writer(csvfile)
            for xpixel in range(H_b0):
                for ypixel in range(H_b0):
                    cpt = xpixel*H_b0+ypixel
                    for enterbase2i in range(3 ** nSize):
                        TT_filter_v2 = []
                        all_TTici = all_TT_vold[0][filter_occurencefunction]
                        all_TT_noiseonlyici = all_TT_noiseonly[0][filter_occurencefunction]
                        dicorefblockfilter = dictionnary_ref[0][filter_occurencefunction]
                        if all_TT_noiseonlyici[enterbase2i]== b'U':
                            #print(all_TTici, all_TT_noiseonlyici, all_TT_noiseonlyici[enterbase2i],enterbase2i, filter_occurencefunction)
                            expressionici = all_TTici[enterbase2i]
                            input_varref, output_binary_ref = dicorefblockfilter[cpt]
                            val_cnf_ici = transform_cnf(input_varref, output_binary_ref,
                                                                 expressionici, [], k=nSize)
                            TT_filter_v2+=val_cnf_ici
                        if len(TT_filter_v2)==0:
                            writer.writerow(["-"])
                        else:
                            #print(TT_filter_v2)
                            writer.writerow(TT_filter_v2)



for filter_occurencefunction in tqdm(range(args.Blocks_filters_output[1])):
    nSize = args.kernel_size_per_block[1] ** 2 * args.groups_per_block[1]
    if (1, filter_occurencefunction) not in nogolist:
        with open(path_save_modelvf +"/TTnet_allposible_block1_filter"+str(filter_occurencefunction)+".csv", 'w') as csvfile:
            writer = csv.writer(csvfile)
            for xpixel in range(H_b1):
                for ypixel in range(H_b1):
                    cpt = xpixel*H_b1+ypixel
                    for enterbase2i in range(3 ** nSize):
                        TT_filter_v2 = []
                        all_TTici = all_TT_vold[1][filter_occurencefunction]
                        all_TT_noiseonlyici = all_TT_noiseonly[1][filter_occurencefunction]
                        dicorefblockfilter = dictionnary_ref[1][filter_occurencefunction]
                        if all_TT_noiseonlyici[enterbase2i]== b'U':
                            expressionici = all_TTici[enterbase2i]
                            input_varref, output_binary_ref = dicorefblockfilter[cpt]
                            val_cnf_ici = transform_cnf(input_varref, output_binary_ref,
                                                                 expressionici, [], k=nSize)
                            TT_filter_v2+=val_cnf_ici
                        if len(TT_filter_v2)==0:
                            writer.writerow(["-"])
                        else:
                            #print(TT_filter_v2)
                            writer.writerow(TT_filter_v2)


