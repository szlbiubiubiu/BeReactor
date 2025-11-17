import torch
# from metric.react2025_me.metric import *

import pdb
clip_size=32


class jiexi:
    def __init__(self):
        dataset_path=''

    


print("-----------------Evaluating Metric-----------------")
# clip_size

is_2025=True
if is_2025:
    from metric.react2025_me.FRC import   compute_FRC_nomp
    from metric.react2025_me.FRD import   compute_FRD_nomp
    from metric.react2025_me.FRDvs import   compute_FRDvs
    from metric.react2025_me.FRVar import   compute_FRVar
    from metric.react2025_me.TLCC import   compute_TLCC_nomp
    # p = args.threads
    # pred: N 10 750 25
    # speaker: N 750 25
    # emo_pred=torch.load("./emo_pred_19.pt").cpu()
    # speaker_emo_gt=torch.load("./speaker_emo_gt_19.pt").cpu()
    # listener_emo_gt=torch.load("./listener_emo_gt_19.pt").cpu()
    # llm_emo_pred=torch.load('./llm_emo_pred_19.pt').cpu()
    emo_pred=torch.load("/data/scratch/acw569/code/react2025_copy/react2025_copy/outputs/motion_transvae/react_2025/offline/251117130041_bsfgcab0/emo_pred_25_base.pt").cpu()[:,:,:,:10]
    speaker_emo_gt=torch.load("/data/scratch/acw569/code/react2025_copy/react2025_copy/outputs/motion_transvae/react_2025/offline/251117130041_bsfgcab0/speaker_emo_gt_25_base.pt").cpu()[:,:,:10]
    listener_emo_gt=torch.load("/data/scratch/acw569/code/react2025_copy/react2025_copy/outputs/motion_transvae/react_2025/offline/251117130041_bsfgcab0/listener_emo_gt_25_base.pt").cpu()[:,:,:10]
    # llm_emo_pred=torch.load('./llm_emo_pred_25_base.pt').cpu()
    # torch.save(all_listener_emotion_pred, "./emo_pred_25_base.pt")
    # torch.save(speaker_emotion_gt, "./speaker_emo_gt_25_base.pt")
    # torch.save(listener_emotion_gt, "./listener_emo_gt_25_base.pt")
    # /data/scratch/acw569/code/react2025_copy/react2025_copy
    pdb.set_trace()
    print('emo_pred.shape',emo_pred.shape)
    print('speaker_emo_gt.shape',speaker_emo_gt.shape)
    print('listener_emo_gt.shape',listener_emo_gt.shape)
    emo_pred_reshape=emo_pred.reshape(clip_size,1,len(emo_pred),25)#10 or 25
    emo_pred_reshape=emo_pred_reshape.permute(2,1,0,3)

    listener_emo_gt_reshape=listener_emo_gt.reshape(clip_size,1,len(emo_pred),25)
    listener_emo_gt_reshape=listener_emo_gt_reshape.permute(2,1,0,3)
    print('emo_pred_reshape.shape',emo_pred_reshape.shape)
    print('speaker_emo_gt.shape',speaker_emo_gt.shape)
    print('listener_emo_gt_reshape.shape',listener_emo_gt_reshape.shape)
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
    pred_shape=all_listener_emotion_pred.shape#[5w,1,1,10]
    all_listener_emotion_pred=all_listener_emotion_pred.reshape(-1,clip_size,pred_shape[2],pred_shape[3])#1612,32,1,10
    all_listener_emotion_pred=all_listener_emotion_pred.permute(0,2,1,3)
    speaker_emo_gt=torch.load("./speaker_emo_gt_24v4.pt").cpu()
    listener_emotion_gt=torch.load("./listener_emo_gt_24v4.pt").cpu()
    listener_emotion_gt_shape=listener_emotion_gt.shape#[5w,1,10]
    listener_emotion_gt=listener_emotion_gt.reshape(-1,clip_size,listener_emotion_gt_shape[2])#1612,32,10
    # listener_emotion_gt=listener_emotion_gt.permute(0,)
    llm_emo_pred=torch.load('./llm_emo_pred_24v4.pt').cpu()
    print('all_listener_emotion_pred.shape',all_listener_emotion_pred.shape)
    print('listener_emotion_gt.shape,',listener_emotion_gt.shape)
    TLCC = compute_TLCC(all_listener_emotion_pred.cpu(), speaker_emo_gt.cpu()[:,:,:10])
    print("TLCC:", TLCC)
    FRC = compute_FRC(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')
    print('FRC:',FRC)
    # If you have problems running function compute_FRD_mp, please replace this function with function compute_FRD
    FRD = compute_FRD(args, all_listener_emotion_pred.cpu(), listener_emotion_gt.cpu(), val_test='test')

    FRDvs = compute_FRDvs(all_listener_emotion_pred.cpu())
    FRVar  = compute_FRVar(all_listener_emotion_pred.cpu())
    smse  = compute_s_mse(all_listener_emotion_pred.cpu())
    # print('TLCC',TLCC)
    print(" FRC: %.4f;FRD: %.4f;\n FRDvs: %.4f;FRVar: %.4f;" % ( FRC, FRD, FRDvs, FRVar))
