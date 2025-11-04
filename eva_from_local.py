import torch
# from metric.react2025_me.metric import *

import pdb
clip_size=32
emo_pred=torch.load("./emo_pred_19.pt").cpu()
speaker_emo_gt=torch.load("./speaker_emo_gt_19.pt").cpu()
listener_emo_gt=torch.load("./listener_emo_gt_19.pt").cpu()
llm_emo_pred=torch.load('./llm_emo_pred_19.pt').cpu()

class jiexi:
    def __init__(self):
        dataset_path=''

    


print("-----------------Evaluating Metric-----------------")
# clip_size

is_2025=False
if is_2025:
    from metric.react2025_me.FRC import   compute_FRC_nomp
    from metric.react2025_me.FRD import   compute_FRD_nomp
    from metric.react2025_me.FRDvs import   compute_FRDvs
    from metric.react2025_me.FRVar import   compute_FRVar
    from metric.react2025_me.TLCC import   compute_TLCC_nomp
    # p = args.threads
    # pred: N 10 750 25
    # speaker: N 750 25
    pdb.set_trace()
    emo_pred_reshape=emo_pred.reshape(len(llm_emo_pred),1,clip_size,10)
    listener_emo_gt_reshape=listener_emo_gt.reshape(len(llm_emo_pred),1,clip_size,10)
    # emo_pred_reshape=listener_emo_gt.reshape(len(llm_emo_pred),clip_size,1,10)
    # If you have problems running function compute_TLCC_mp, please replace this function with function compute_TLCC
    # TLCC = compute_TLCC_nomp(emo_pred_reshape, speaker_emo_gt[:,:,-10:])

    # If you have problems running function compute_FRC_mp, please replace this function with function compute_FRC
    FRC = compute_FRC_nomp(emo_pred_reshape, listener_emo_gt_reshape)
    print("FRC:",FRC)
    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD_nomp(emo_pred_reshape, listener_emo_gt_reshape)
    print("FRD:",FRD)
    # FRDvs = compute_FRDvs(all_listener_emotion_pred)
    FRVar  = compute_FRVar(emo_pred_reshape)
    # smse  = compute_s_mse(emo_pred_reshape)
    print("FRVar:",FRVar)
    print("Metric: | FRC: {:.5f} | FRD: {:.5f} | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} | TLCC: {:.5f}".format(FRC, FRD, smse, FRVar, FRDvs, TLCC))
    print("Latex-friendly --> model_name & {:.2f} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & - & {:.2f} \\\\".format( FRC, FRD, smse, FRVar, FRDvs, TLCC))


    #evaluate language
    listener_emo_gt_reshape=listener_emo_gt.reshape(len(llm_emo_pred),clip_size,1,10)
    language_mse=torch.nn.functional.mse_loss(llm_emo_pred,listener_emo_gt_reshape)#check listener_emotion_gt.shape
    print('language_mse',language_mse)

else:
    from metric import *
    args=jiexi()
    args.dataset_path='./data'

    all_listener_emotion_pred=torch.load("./emo_pred_24v4.pt").cpu()
    pred_shape=all_listener_emotion_pred.shape
    all_listener_emotion_pred=all_listener_emotion_pred.reshape(clip_size,-1,pred_shape[1],pred_shape[2],pred_shape[3])
    all_listener_emotion_pred=all_listener_emotion_pred.transpose(0,1)
    speaker_emo_gt=torch.load("./speaker_emo_gt_24v4.pt").cpu()
    listener_emotion_gt=torch.load("./listener_emo_gt_24v4.pt").cpu()
    listener_emotion_gt_shape=listener_emotion_gt.shape
    listener_emotion_gt=listener_emotion_gt.reshape(clip_size,-1,listener_emotion_gt_shape[1],listener_emotion_gt_shape[2])
    listener_emotion_gt=listener_emotion_gt.transpose(0,1)
    llm_emo_pred=torch.load('./llm_emo_pred_24v4.pt').cpu()
    print('all_listener_emotion_pred.shape',all_listener_emotion_pred.shape)
    print('listener_emotion_gt.shape,',listener_emotion_gt.shape)
    FRC = compute_FRC(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')
    print('FRC:',FRC)
    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

    FRDvs = compute_FRDvs(all_listener_emotion_pred.cpu())
    FRVar  = compute_FRVar(all_listener_emotion_pred.cpu())
    smse  = compute_s_mse(all_listener_emotion_pred.cpu())
    # print('TLCC',TLCC)
    print("TLCC: %.4f; FRC: %.4f;FRD: %.4f;\n FRDvs: %.4f;FRVar: %.4f;smse: %.4f;" % (TLCC, FRC, FRD, FRDvs, FRVar, smse))
