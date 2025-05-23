import torch
import numpy as np
from networks.freqnet import freqnet
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, f1_score
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_labels = (y_pred > 0.5).astype(int)

    # Counts
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()

    # Correct predictions
    correct_real = ((y_true == 0) & (y_pred_labels == 0)).sum()
    correct_fake = ((y_true == 1) & (y_pred_labels == 1)).sum()

    # Per-class accuracy
    real_acc = correct_real / num_real if num_real > 0 else np.nan
    fake_acc = correct_fake / num_fake if num_fake > 0 else np.nan

    # Overall accuracy
    acc = (correct_real + correct_fake) / (num_real + num_fake)

    # Average precision and F1 score
    ap = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_labels)

    # Balanced accuracy (optional): mean of real_acc and fake_acc
    balanced_acc = np.nanmean([real_acc, fake_acc])

    return acc, ap, real_acc, fake_acc, f1, balanced_acc, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = freqnet(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, real_acc, fake_acc, f1, balanced_acc, _ = validate(model, opt)

    print("Accuracy:", acc)
    print("Average Precision:", ap)
    print("Real Accuracy:", real_acc)
    print("Fake Accuracy:", fake_acc)
    print("F1 Score:", f1)
    print("Balanced Accuracy:", balanced_acc)