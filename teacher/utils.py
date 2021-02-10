#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import logging
from logging import handlers
import numpy as np
import torch.nn.functional as F



def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size.
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0),
                            enc_hiddens.size(1),
                            dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def validate(model, dataloader, device):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    model.to(device)
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments,
             batch_labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            logits = model(seqs, masks, segments)
            loss = compute_loss(labels, logits)
            running_loss += loss.item()
            running_accuracy += correct_predictions(logits, labels)
            all_prob.extend(logits[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(
        all_labels, all_prob)


def test(model, dataloader, device):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    model.to(device)
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments,
             batch_labels) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(
                device), batch_seq_masks.to(device), batch_seq_segments.to(
                device), batch_labels.to(device)
            logits = model(seqs, masks, segments)
            accuracy += correct_predictions(logits, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(logits[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(
        all_labels, all_prob)


def train(args, model, dataloader, optimizer, epoch_number, max_gradient_norm, logger, device):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    model.to(device)
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    total_batch = 0
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments,
                      batch_labels) in enumerate(dataloader):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        seqs, masks, segments, labels = batch_seqs.to(
            device), batch_seq_masks.to(device), batch_seq_segments.to(
            device), batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids=seqs,
                       attention_mask=masks,
                       token_type_ids=segments)
        loss = compute_loss(labels, logits)  # 计算损失
        # loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        # optimizer.step()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if (batch_index + 1) % args.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            optimizer.zero_grad()
        batch_time_avg += time.time() - batch_start
        correct_pred = correct_predictions(logits, labels)
        if total_batch % 200 == 0:
            description = "Batch:{}  time: {:.2f}s loss: {:.4f}  accuracy: {:.4f}" \
                .format(total_batch, batch_time_avg, loss.item(), correct_pred / len(batch_labels))
            logger.info(description)
        correct_preds += correct_pred
        running_loss += loss.item()
        total_batch += 1
    if args.swa:
        optimizer.swap_swa_sgd()
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def create_logger(log_path):
    """
    日志的创建
    :param log_path:
    :return:
    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # F范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def compute_loss(labels, logits):
    """计算交叉熵损失"""
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    return loss


def compute_f1(logits, labels):
    out = logits[:, 0].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    f1 = f1_score(labels, out, zero_division=1)
    # R = out.sum()
    # P = labels.sum()
    # TP += ((out + labels) > 1).sum()i
    # pre = TP / R
    # rec = TP / P
    # return 2 * (pre * rec) / (pre + rec)
    return f1
    # accuracy += correct_predictions(probabilities, labels)
    # probs = out_classes.cpu().numpy()
    # labels = batch_labels.detach().cpu().numpy()
