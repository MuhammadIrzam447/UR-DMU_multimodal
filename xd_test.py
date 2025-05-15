import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('comet_ml').setLevel(logging.WARNING)
def test(net, config, experiment, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/xd_gt.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []

        for i in range(len(test_loader.dataset) // 5):
            batch = next(load_iter)

            # Check if batch contains RGB, Audio, and label
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                rgb_data, audio_data, label = batch
                rgb_data = rgb_data.cuda()
                audio_data = audio_data.cuda()
                label = label.cuda()

                cls_label.append(int(label[0]))

                # Forward pass for both modalities
                rgb_res = net(rgb_data)
                audio_res = net(audio_data)

                # Frame predictions
                rgb_predict = rgb_res["frame"].cpu().numpy().mean(0)
                audio_predict = audio_res["frame"].cpu().numpy().mean(0)

                # Combine predictions
                combined_predict = (rgb_predict + audio_predict) / 2.0

                # Binary classification decision
                cls_pre.append(1 if combined_predict.max() > 0.5 else 0)

                # Expand to frame level (repetition)
                fpre_ = np.repeat(combined_predict, 16)

                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])

            else:
                # Fallback: original single input (only RGB or video input)
                data, label = batch
                data = data.cuda()
                label = label.cuda()

                cls_label.append(int(label[0]))

                res = net(data)
                a_predict = res["frame"].cpu().numpy().mean(0)

                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)

                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
      
        corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = corrent_num / (len(cls_pre))
       
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)
      
        # wind.plot_lines('roc_auc', auc_score)
        # wind.plot_lines('accuracy', accuracy)
        # wind.plot_lines('pr_auc', ap_score)
        # wind.lines('scores', frame_predict)
        # wind.lines('roc_curve', tpr, fpr)
        # Log metrics to Comet ML
        # Log scalar metrics
        experiment.log_metric("roc_auc", auc_score, step=step)
        experiment.log_metric("accuracy", accuracy, step=step)
        experiment.log_metric("pr_auc", ap_score, step=step)

        # Log ROC curve
        experiment.log_curve(name="ROC Curve", x=fpr.tolist(), y=tpr.tolist(), step=step)
        # Log Precision-Recall curve
        experiment.log_curve(name="Precision-Recall Curve", x=recall.tolist(), y=precision.tolist(), step=step)
        # Log prediction scores as histogram
        experiment.log_histogram_3d(values=frame_predict.tolist(), name="Prediction Scores", step=step)

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        