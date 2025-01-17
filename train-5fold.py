import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import KFold

import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import shutil
import logging
from tqdm import tqdm
from datetime import datetime
import argparse

from dataload import load_data, create_loaders
from model_dict.GCN_basic import GcnNet
from model_dict.GATN import GAT

warnings.filterwarnings("ignore")
model_dict = {
    "GCN_basic": GcnNet(n_input=20),
    "GAT": GAT(n_input=20, nclass=2, nheads=3, dropout=0.5)
}


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=9067, help='Random seed.')
    parser.add_argument('--dataset_path', type=str, default="./data/postprocess_data")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.025523868808165096)
    parser.add_argument('--weight_decay', type=float, default=0.00011842653778790806,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--batch_size", type=int, default=19)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default="GCN_basic")
    parser.add_argument("--patience", type=int, default=1000000,
                        help='Threshold of early stop')

    args = parser.parse_args()
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log_folder = logger()
    args.log_folder = log_folder
    logging.info(f"model:{args.model_name}  epochs:{args.epochs}  lr:{args.lr}  patience:{args.patience}")
    return args


def logger():
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    log_folder = f"./logs/{current_time}"
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "log.txt")
    logging.basicConfig(
        level=logging.INFO,  # 日志级别为 INFO
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_folder


def draw_png(args, fpr_list, tpr_list, roc_auc_list, epoch_train_losses):
    # 设置全局字体（使用系统字体 Arial 或 Times New Roman）
    rcParams['font.sans-serif'] = ['Times New Roman']  # 替换为 'Times New Roman' 或其他字体名称
    rcParams['axes.unicode_minus'] = False  # 正常显示负号

    mean_fpr = np.linspace(0, 1, 100)

    # 绘制每一折的AUC曲线和五折平均AUC曲线
    plt.figure(figsize=(10, 6))
    for i, (fpr, tpr, roc_auc) in enumerate(zip(fpr_list, tpr_list, roc_auc_list)):
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')

    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='-', linewidth=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')
    x1 = np.arange(0, 1.1, 0.1)
    plt.plot(x1, x1, linestyle='--', color='blue', lw=2)
    x = np.arange(0, 1.1, 0.2)
    y = np.arange(0, 1.1, 0.2)
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for 5-Fold Cross Validation', fontsize=12)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(False)
    plt.savefig(os.path.join(args.log_folder, "roc_curves.png"))

    # 绘制五折平均训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), np.mean(epoch_train_losses, axis=0), label='Mean Train Loss')
    plt.title('Mean Training Loss per Epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.savefig(
        os.path.join(args.log_folder, "mean_train_loss_per_epoch.png"))
    # plt.show()


def train(args, model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    all_labels = []
    all_probs = []

    for feature, adj_matrix, label in train_loader:
        feature, adj_matrix, label = feature.to(args.device), adj_matrix.to(args.device), label.to(args.device)

        optimizer.zero_grad()
        output = model(adj_matrix, feature)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        probs = torch.softmax(output, dim=1)
        all_labels.extend(label.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy()[:, 1])

    return epoch_loss / len(train_loader), all_labels, all_probs


def evaluate(args, model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for feature, adj_matrix, label in val_loader:
            feature, adj_matrix, label = feature.to(args.device), adj_matrix.to(args.device), label.to(args.device)

            output = model(adj_matrix, feature)
            loss = criterion(output, label)

            val_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy()[:, 1])

    return val_loss / len(val_loader), all_labels, all_probs


def train_val(args):
    # 分成 n_splits 块，且随机打乱
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold = 0

    train_auc_scores = []
    val_auc_scores = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1_scores = []
    val_f1_scores = []
    train_specificities = []
    val_specificities = []

    epoch_train_losses = np.zeros((5, args.epochs))

    fpr_list = []
    tpr_list = []
    roc_auc_list = []

    dataset = load_data(args.dataset_path)
    for train_index, val_index in kf.split(dataset):
        fold += 1
        train_loader, val_loader = create_loaders(dataset, train_index, val_index, batch_size=args.epochs)

        try:
            model = model_dict[args.model_name].to(args.device)
        except Exception as _:
            shutil.rmtree(args.log_folder)
            logging.error(f"Model '{args.model_name}' is not defined in model_dict.")
            exit()

        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        patience_counter = 0
        best_val_auc = 0
        best_val_acc = 0
        best_fpr = None
        best_tpr = None

        for epoch in tqdm(range(args.epochs), "Training process:"):
            train_loss, train_labels, train_probs = train(args, model, train_loader, criterion, optimizer)
            val_loss, val_labels, val_probs = evaluate(args, model, val_loader, criterion)

            epoch_train_losses[fold - 1, epoch] = train_loss

            val_auc = roc_auc_score(val_labels, val_probs)
            val_acc = accuracy_score(val_labels, np.round(val_probs))

            # 如果AUC或ACC有提升，重置耐心计数器，否则增加计数器
            if val_auc > best_val_auc or val_acc > best_val_acc:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    fpr, tpr, _ = roc_curve(val_labels, val_probs, pos_label=1)
                    best_fpr = fpr
                    best_tpr = tpr
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logging.info(f"Early stopping at epoch {epoch + 1} for fold {fold}")
                break

        fpr_list.append(best_fpr)
        tpr_list.append(best_tpr)
        roc_auc_list.append(best_val_auc)

        train_auc = roc_auc_score(train_labels, train_probs)
        train_acc = accuracy_score(train_labels, np.round(train_probs))
        train_precision = precision_score(train_labels, np.round(train_probs))
        val_precision = precision_score(val_labels, np.round(val_probs))
        train_recall = recall_score(train_labels, np.round(train_probs))
        val_recall = recall_score(val_labels, np.round(val_probs))
        train_f1 = f1_score(train_labels, np.round(train_probs))
        val_f1 = f1_score(val_labels, np.round(val_probs))

        tn, fp, fn, tp = confusion_matrix(train_labels, np.round(train_probs)).ravel()
        train_specificity = tn / (tn + fp)
        tn, fp, fn, tp = confusion_matrix(val_labels, np.round(val_probs)).ravel()
        val_specificity = tn / (tn + fp)

        train_auc_scores.append(train_auc)
        val_auc_scores.append(best_val_auc)
        train_accuracies.append(train_acc)
        val_accuracies.append(best_val_acc)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        train_specificities.append(train_specificity)
        val_specificities.append(val_specificity)

        logging.info(f"Fold {fold}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"Train AUC: {train_auc:.4f}, Val AUC: {best_val_auc:.4f}")
        logging.info(f"Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}")
        logging.info(f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")
        logging.info(f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}")
        logging.info(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        logging.info(f"Train Specificity: {train_specificity:.4f}, Val Specificity: {val_specificity:.4f} \n")

    # 计算和输出五折平均值
    logging.info(f"Mean Train AUC: {np.mean(train_auc_scores):.4f}, "
                 f"Mean Val AUC: {np.mean(val_auc_scores):.4f}")
    logging.info(f"Mean Train Acc: {np.mean(train_accuracies):.4f}, "
                 f"Mean Val Acc: {np.mean(val_accuracies):.4f}")
    logging.info(f"Mean Train Precision: {np.mean(train_precisions):.4f}, "
                 f"Mean Val Precision: {np.mean(val_precisions):.4f}")
    logging.info(f"Mean Train Recall: {np.mean(train_recalls):.4f}, "
                 f"Mean Val Recall: {np.mean(val_recalls):.4f}")
    logging.info(f"Mean Train F1: {np.mean(train_f1_scores):.4f},"
                 f" Mean Val F1: {np.mean(val_f1_scores):.4f}")
    logging.info(f"Mean Train Specificity: {np.mean(train_specificities):.4f}, "
                 f"Mean Val Specificity: {np.mean(val_specificities):.4f}")

    return model, fpr_list, tpr_list, roc_auc_list, epoch_train_losses


if __name__ == '__main__':
    args = args_parse()
    model, fpr_list, tpr_list, roc_auc_list, epoch_train_losses = train_val(args)
    draw_png(args, fpr_list, tpr_list, roc_auc_list, epoch_train_losses)

    # 保存模型参数
    check_path = os.path.join(args.log_folder, "check_point")
    os.makedirs(check_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(check_path, "model.pth"))
