# utils/evaluation.py
"""
模型评估工具
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from .angle_utils import compute_circular_correlation, compute_mae_degrees

logger = logging.getLogger(__name__)

def evaluate_model(model, data_loader, device, torsion_types, output_dir=None):
    model.eval()
    
    # 初始化结果字典
    all_predictions = {angle: [] for angle in torsion_types}
    all_targets = {angle: [] for angle in torsion_types}
    all_masks = {angle: [] for angle in torsion_types}
    all_seq_lens = {angle: [] for angle in torsion_types}  # 跟踪序列长度
    pdb_ids = []
    chain_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            tokens = batch['tokens'].to(device)
            pdb_ids.extend(batch['pdb_ids'])
            chain_ids.extend(batch['chain_ids'])
            
            # 获取预测
            predictions, _ = model(tokens)


            # 处理每种角度类型
            for angle_name in torsion_types:
                if angle_name in batch['angles']:
                    angle_target = batch['angles'][angle_name].to(device)
                    angle_mask = batch['masks'][angle_name].to(device)
                    pred = predictions[angle_name]
                    
                    # 分别处理批次中的每个序列
                    batch_size = angle_target.shape[0]
                    for i in range(batch_size):
                        target_len = angle_target[i].shape[0]
                        pred_len = pred[i].shape[0]
                        min_len = min(target_len, pred_len)
                        
                        # 存储单个序列的预测和目标
                        all_predictions[angle_name].append(pred[i, :min_len].unsqueeze(0))
                        all_targets[angle_name].append(angle_target[i, :min_len].unsqueeze(0))
                        all_masks[angle_name].append(angle_mask[i, :min_len].unsqueeze(0))
                        all_seq_lens[angle_name].append(min_len)
    
    # 计算指标
    metrics = {}

    # 添加直接打印语句，确保始终输出
    print("\n===== 评估指标 =====")
    
    for angle_name in torsion_types:
        if all_predictions[angle_name]:
            # 分别计算每个序列的指标并平均
            mae_values = []
            corr_values = []
            
            # 在evaluate_model函数中，替换这部分代码
            for i in range(len(all_predictions[angle_name])):
                pred = all_predictions[angle_name][i].squeeze(0)  # 移除批次维度
                target = all_targets[angle_name][i].squeeze(0)    # 移除批次维度
                mask = all_masks[angle_name][i].squeeze(0)        # 移除批次维度
                
                # 确保是NumPy数组
                pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
                target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                
                # 计算指标
                try:
                    mae = compute_mae_degrees(pred_np, target_np, mask_np)
                    corr = compute_circular_correlation(pred_np, target_np, mask_np)
                    
                    if not np.isnan(mae) and not np.isinf(mae) and mae is not None:
                        mae_values.append(mae)
                    if not np.isnan(corr) and not np.isinf(corr) and corr is not None:
                        corr_values.append(corr)
                except Exception as e:
                    logger.warning(f"计算{angle_name}指标时发生错误: {str(e)}")

            if mae_values:
                avg_mae = np.mean(mae_values)
                metrics[f"{angle_name}_mae"] = avg_mae
                print(f"{angle_name} MAE: {avg_mae:.2f}°")
            else:
                print(f"{angle_name} MAE: 无有效数据")
                
            if corr_values:
                avg_corr = np.mean(corr_values)
                metrics[f"{angle_name}_corr"] = avg_corr
                print(f"{angle_name} 循环相关: {avg_corr:.4f}")
            else:
                print(f"{angle_name} 循环相关: 无有效数据")
                
            # 同时保留现有的logger输出
            logger.info(f"{angle_name} - MAE: {metrics.get(f'{angle_name}_mae', 'N/A')}, 循环相关性: {metrics.get(f'{angle_name}_corr', 'N/A')}")
    
    # 计算并打印总体指标
    angle_metrics = [(angle, metrics.get(f"{angle}_mae", np.nan)) for angle in torsion_types]
    valid_metrics = [(angle, value) for angle, value in angle_metrics if not np.isnan(value)]
    
    if valid_metrics:
        avg_mae = np.mean([value for _, value in valid_metrics])
        metrics["avg_mae"] = avg_mae
        print(f"总体平均 MAE: {avg_mae:.2f}°")
        logger.info(f"平均 MAE: {avg_mae:.2f}°")
    else:
        print("无法计算平均 MAE (没有有效指标)")
    
    print("========================\n")
    
    
    # 如果提供了输出目录则保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        
        # 其他可视化代码保持不变
        
    return metrics