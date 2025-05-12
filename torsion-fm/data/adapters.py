# data/adapters.py
import logging
import numpy as np
from collections import defaultdict
from .preprocessing import calculate_dihedral

logger = logging.getLogger(__name__)

def compute_torsion_angles_single(atom_coords_dict, sorted_residue_ids):
    """计算单个RNA的扭转角"""
    # 定义扭转角所需的原子
    torsion_definitions = {
        'alpha': ["O3'", 'P', "O5'", "C5'"],    # O3'(i-1)-P(i)-O5'(i)-C5'(i)
        'beta': ['P', "O5'", "C5'", "C4'"],     # P(i)-O5'(i)-C5'(i)-C4'(i)
        'gamma': ["O5'", "C5'", "C4'", "C3'"],  # O5'(i)-C5'(i)-C4'(i)-C3'(i)
        'delta': ["C5'", "C4'", "C3'", "O3'"],  # C5'(i)-C4'(i)-C3'(i)-O3'(i)
        'epsilon': ["C4'", "C3'", "O3'", 'P'],  # C4'(i)-C3'(i)-O3'(i)-P(i+1)
        'zeta': ["C3'", "O3'", 'P', "O5'"],     # C3'(i)-O3'(i)-P(i+1)-O5'(i+1)
        'chi': ["O4'", "C1'", 'N9', 'C4']       # O4'(i)-C1'(i)-N9(i)-C4(i) (嘌呤)
    }
    
    # 嘧啶的chi角替代定义
    chi_pyrimidine = ["O4'", "C1'", 'N1', 'C2']  # O4'(i)-C1'(i)-N1(i)-C2(i) (嘧啶)
    
    # 初始化结果
    torsion_angles = {angle_name: [] for angle_name in torsion_definitions}
    torsion_masks = {angle_name: [] for angle_name in torsion_definitions}
    
    # 对每个残基计算扭转角
    for i, res_id in enumerate(sorted_residue_ids):
        # 当前残基的原子坐标
        current_res_atoms = atom_coords_dict.get(res_id, {})
        
        # 获取当前残基的碱基类型(如果可用)
        residue_name = None
        if isinstance(current_res_atoms, dict) and 'residue_name' in current_res_atoms:
            residue_name = current_res_atoms['residue_name']
            current_res_atoms = current_res_atoms['atom_coords']
        
        # 计算各种扭转角
        for angle_name, atom_names in torsion_definitions.items():
            angle = None
            mask = 0
            
            try:
                # 根据角度类型选择相应的原子
                if angle_name == 'alpha' and i > 0:
                    # alpha需要前一个残基的O3'
                    prev_res_id = sorted_residue_ids[i-1]
                    prev_res_atoms = atom_coords_dict.get(prev_res_id, {})
                    if isinstance(prev_res_atoms, dict) and 'residue_name' in prev_res_atoms:
                        prev_res_atoms = prev_res_atoms['atom_coords']
                    
                    if "O3'" in prev_res_atoms and all(atom in current_res_atoms for atom in atom_names[1:]):
                        p1 = prev_res_atoms["O3'"]
                        p2 = current_res_atoms['P']
                        p3 = current_res_atoms["O5'"]
                        p4 = current_res_atoms["C5'"]
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == 'epsilon' and i < len(sorted_residue_ids) - 1:
                    # epsilon需要下一个残基的P
                    next_res_id = sorted_residue_ids[i+1]
                    next_res_atoms = atom_coords_dict.get(next_res_id, {})
                    if isinstance(next_res_atoms, dict) and 'residue_name' in next_res_atoms:
                        next_res_atoms = next_res_atoms['atom_coords']
                    
                    if 'P' in next_res_atoms and all(atom in current_res_atoms for atom in atom_names[:3]):
                        p1 = current_res_atoms["C4'"]
                        p2 = current_res_atoms["C3'"]
                        p3 = current_res_atoms["O3'"]
                        p4 = next_res_atoms['P']
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == 'zeta' and i < len(sorted_residue_ids) - 1:
                    # zeta需要下一个残基的P和O5'
                    next_res_id = sorted_residue_ids[i+1]
                    next_res_atoms = atom_coords_dict.get(next_res_id, {})
                    if isinstance(next_res_atoms, dict) and 'residue_name' in next_res_atoms:
                        next_res_atoms = next_res_atoms['atom_coords']
                    
                    if all(atom in next_res_atoms for atom in atom_names[2:]) and all(atom in current_res_atoms for atom in atom_names[:2]):
                        p1 = current_res_atoms["C3'"]
                        p2 = current_res_atoms["O3'"]
                        p3 = next_res_atoms['P']
                        p4 = next_res_atoms["O5'"]
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
                
                elif angle_name == 'chi':
                    # 根据碱基类型选择chi角度定义
                    if residue_name in ['A', 'G']:  # 嘌呤
                        if all(atom in current_res_atoms for atom in ["O4'", "C1'", 'N9', 'C4']):
                            p1 = current_res_atoms["O4'"]
                            p2 = current_res_atoms["C1'"]
                            p3 = current_res_atoms['N9']
                            p4 = current_res_atoms['C4']
                            angle = calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                    elif residue_name in ['C', 'U']:  # 嘧啶
                        if all(atom in current_res_atoms for atom in ["O4'", "C1'", 'N1', 'C2']):
                            p1 = current_res_atoms["O4'"]
                            p2 = current_res_atoms["C1'"]
                            p3 = current_res_atoms['N1']
                            p4 = current_res_atoms['C2']
                            angle = calculate_dihedral(p1, p2, p3, p4)
                            if angle is not None:
                                mask = 1
                
                else:  # beta, gamma, delta等只需要当前残基的原子
                    if all(atom in current_res_atoms for atom in atom_names):
                        p1 = current_res_atoms[atom_names[0]]
                        p2 = current_res_atoms[atom_names[1]]
                        p3 = current_res_atoms[atom_names[2]]
                        p4 = current_res_atoms[atom_names[3]]
                        angle = calculate_dihedral(p1, p2, p3, p4)
                        if angle is not None:
                            mask = 1
            except Exception as e:
                logger.debug(f"计算{angle_name}角度失败: {str(e)}")
            
            # 将结果添加到列表
            torsion_angles[angle_name].append(angle if angle is not None else 0.0)
            torsion_masks[angle_name].append(mask)
    
    return torsion_angles, torsion_masks

def adapt_training_dict_single(data):
    """
    适配Training_Dict_single.pkl格式的数据
    
    参数:
        data: 原始数据字典，键为结构ID，值为RNA结构信息
    
    返回:
        list: 转换后的数据列表，符合模型预期的格式
    """
    adapted_data = []
    
    # 处理每个RNA结构
    for key, item in data.items():
        try:
            # 检查格式是否正确
            if not isinstance(item, dict) or 'rna_dic' not in item:
                logger.warning(f"结构 {key} 不符合预期格式，跳过")
                continue
            
            # 提取基本信息
            pdb_id = item.get('pdb_id', f'unknown_{key}')
            chain_id = item.get('chain_id', 'A')
            is_multi_chain = item.get('if_multi_chain', False)
            
            # 跳过多链RNA
            if is_multi_chain:
                logger.info(f"跳过多链RNA: {pdb_id}")
                continue
            
            # 获取RNA字典
            rna_dic = item['rna_dic']
            if not isinstance(rna_dic, dict) or len(rna_dic) == 0:
                logger.warning(f"RNA字典为空或格式无效: {pdb_id}，跳过")
                continue
            
            # 排序残基ID确保序列顺序正确
            sorted_residue_ids = sorted(rna_dic.keys())
            
            # 提取序列和原子坐标
            sequence = ""
            for res_id in sorted_residue_ids:
                residue = rna_dic[res_id]
                
                # 检查残基格式
                if not isinstance(residue, dict) or 'residue_name' not in residue:
                    logger.warning(f"残基 {res_id} 在 {pdb_id} 中格式无效，跳过")
                    continue
                
                residue_name = residue['residue_name']
                sequence += residue_name
            
            # 计算扭转角
            torsion_angles, torsion_masks = compute_torsion_angles_single(rna_dic, sorted_residue_ids)
            
            # 创建适配后的结构
            adapted_item = {
                'pdb_id': pdb_id,
                'chain_id': chain_id,
                'sequence': sequence,
                'torsion_angles': torsion_angles,
                'torsion_masks': torsion_masks,
                'sorted_residue_ids': sorted_residue_ids
            }
            
            adapted_data.append(adapted_item)
            logger.debug(f"成功处理结构: {pdb_id}, 序列长度: {len(sequence)}")
            
        except Exception as e:
            logger.warning(f"处理结构 {key} 时出错: {str(e)}")
    
    logger.info(f"适配转换: 原始数据 {len(data)} 项 -> 适配后 {len(adapted_data)} 项")
    return adapted_data