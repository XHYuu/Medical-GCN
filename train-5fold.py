import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from dataload import load_data, create_loaders
from model import GcnNet
from matplotlib import rcParams

warnings.filterwarnings("ignore")

current_time = datetime.now().strftime("%m%d_%H%M%S")  # 格式化时间
log_folder = f"logs/{current_time}"  # 日志文件夹路径
os.makedirs(log_folder, exist_ok=True)  # 创建日志文件夹
log_file = os.path.join(log_folder, "log.txt")
logging.basicConfig(
    level=logging.INFO,  # 日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    datefmt="%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),  # 日志输出到文件
        logging.StreamHandler()         # 日志同时输出到控制台
    ]
)

logging.info("basic_GCN \n")

# 超参数设置
LEARNING_RATE = 0.025523868808165096
WEIGHT_DECAY = 0.00011842653778790806
EPOCHS = 1000
BATCH_SIZE = 19
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 9067
PATIENCE = 20  # 早停的耐心值

# 加载数据
main_path = './data/postprocess_data'
dataset = load_data(main_path)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 定义模型、损失函数和优化器
input_dim = 20


def train(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    all_labels = []
    all_probs = []

    for feature, adj_matrix, label in train_loader:
        feature, adj_matrix, label = feature.to(DEVICE), adj_matrix.to(DEVICE), label.to(DEVICE)

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


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for feature, adj_matrix, label in val_loader:
            feature, adj_matrix, label = feature.to(DEVICE), adj_matrix.to(DEVICE), label.to(DEVICE)

            output = model(adj_matrix, feature)
            loss = criterion(output, label)

            val_loss += loss.item()
            probs = torch.softmax(output, dim=1)
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy()[:, 1])

    return val_loss / len(val_loader), all_labels, all_probs


kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
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

epoch_train_losses = np.zeros((5, EPOCHS))

fpr_list = []
tpr_list = []
roc_auc_list = []
mean_fpr = np.linspace(0, 1, 100)

for train_index, val_index in kf.split(dataset):
    fold += 1
    train_loader, val_loader = create_loaders(dataset, train_index, val_index, batch_size=BATCH_SIZE)

    model = GcnNet(input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    patience_counter = 0
    best_val_auc = 0
    best_val_acc = 0
    best_fpr = None
    best_tpr = None

    for epoch in range(EPOCHS):
        train_loss, train_labels, train_probs = train(model, train_loader, criterion, optimizer)
        val_loss, val_labels, val_probs = evaluate(model, val_loader, criterion)

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

        if patience_counter >= PATIENCE:
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
mean_train_auc = np.mean(train_auc_scores)
mean_val_auc = np.mean(val_auc_scores)
mean_train_acc = np.mean(train_accuracies)
mean_val_acc = np.mean(val_accuracies)
mean_train_precision = np.mean(train_precisions)
mean_val_precision = np.mean(val_precisions)
mean_train_recall = np.mean(train_recalls)
mean_val_recall = np.mean(val_recalls)
mean_train_f1 = np.mean(train_f1_scores)
mean_val_f1 = np.mean(val_f1_scores)
mean_train_specificity = np.mean(train_specificities)
mean_val_specificity = np.mean(val_specificities)

logging.info(f"Mean Train AUC: {mean_train_auc:.4f}, Mean Val AUC: {mean_val_auc:.4f}")
logging.info(f"Mean Train Acc: {mean_train_acc:.4f}, Mean Val Acc: {mean_val_acc:.4f}")
logging.info(f"Mean Train Precision: {mean_train_precision:.4f}, Mean Val Precision: {mean_val_precision:.4f}")
logging.info(f"Mean Train Recall: {mean_train_recall:.4f}, Mean Val Recall: {mean_val_recall:.4f}")
logging.info(f"Mean Train F1: {mean_train_f1:.4f}, Mean Val F1: {mean_val_f1:.4f}")
logging.info(f"Mean Train Specificity: {mean_train_specificity:.4f}, Mean Val Specificity: {mean_val_specificity:.4f}")

# 设置全局字体（使用系统字体 Arial 或 Times New Roman）
rcParams['font.sans-serif'] = ['Times New Roman']  # 替换为 'Times New Roman' 或其他字体名称
rcParams['axes.unicode_minus'] = False  # 正常显示负号

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
plt.xlabel('False Positive Rate', fontsize=12)  # 修改 x 轴标签字体大小
plt.ylabel('True Positive Rate', fontsize=12)  # 修改 y 轴标签字体大小
plt.title('ROC Curves for 5-Fold Cross Validation', fontsize=12)  # 修改标题字体大小
plt.tick_params(axis='both', labelsize=14)  # 修改刻度字体大小
plt.legend(loc="lower right", fontsize=14)  # 修改图例字体大小
plt.grid(False)
plt.savefig(os.path.join(log_folder, "roc_curves.png"))
# plt.show()

# 绘制五折平均训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(EPOCHS), np.mean(epoch_train_losses, axis=0), label='Mean Train Loss')
plt.title('Mean Training Loss per Epoch', fontsize=20)  # 修改标题字体大小
plt.xlabel('Epoch', fontsize=20)  # 修改 x 轴标签字体大小
plt.ylabel('Loss', fontsize=20)  # 修改 y 轴标签字体大小
plt.tick_params(axis='both', labelsize=18)  # 修改刻度字体大小
plt.legend(fontsize=18)  # 修改图例字体大小
plt.grid()
plt.savefig(
    os.path.join(log_folder, "mean_train_loss_per_epoch.png"))
# plt.show()
