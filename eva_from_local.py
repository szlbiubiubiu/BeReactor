import torch
# from metric.react2025_me.metric import *
from metric.react2025_me.FRC import   compute_FRC
from metric.react2025_me.FRD import   compute_FRD
from metric.react2025_me.FRDvs import   compute_FRDvs
from metric.react2025_me.FRVar import   compute_FRVar
from metric.react2025_me.TLCC import   compute_TLCC
import multiprocessing


multiprocessing.set_start_method('spawn')
clip_size=32
emo_pred=torch.load("./emo_pred.pt")
speaker_emo_gt=torch.load("./speaker_emo_gt.pt")
listener_emo_gt=torch.load("./listener_emo_gt.pt")
llm_emo_pred=torch.load('./llm_emo_pred.pt')

print("-----------------Evaluating Metric-----------------")
# clip_size

# p = args.threads

# If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
TLCC = compute_TLCC(emo_pred, speaker_emo_gt)

# If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
FRC = compute_FRC(emo_pred, listener_emo_gt, val_test='val')

# If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
FRD = compute_FRD(emo_pred, listener_emo_gt, val_test='val')

# FRDvs = compute_FRDvs(all_listener_emotion_pred)
FRVar  = compute_FRVar(emo_pred)
smse  = compute_s_mse(emo_pred)

print("Metric: | FRC: {:.5f} | FRD: {:.5f} | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | TLCC: {:.5f}".format(FRC, FRD, smse, FRVar, FRDvs, TLCC))
print("Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format( FRC, FRD, smse, FRVar, FRDvs, TLCC))


#evaluate language
listener_emo_gt_reshape=listener_emo_gt.reshape(len(llm_emo_pred),clip_size,1,10)
language_mse=torch.nn.functional.mse_loss(llm_emo_pred,listener_emo_gt_reshape)#check listener_emotion_gt.shape
print('language_mse',language_mse)