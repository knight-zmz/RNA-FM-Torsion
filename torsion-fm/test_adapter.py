# test_adapter.py
import os
import pickle
import logging
import torch
from data.adapters import adapt_training_dict_single

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_adapter(pkl_file, output_dir=None):
    """测试数据适配器"""
    logger.info(f"测试适配器: {pkl_file}")
    
    # 加载PKL文件
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"加载了数据，包含 {len(data)} 个结构")
    
    # 输出第一个结构的信息
    first_key = list(data.keys())[0]
    first_item = data[first_key]
    logger.info(f"第一个结构 (键={first_key}):")
    logger.info(f"  PDB ID: {first_item.get('pdb_id', 'unknown')}")
    logger.info(f"  链 ID: {first_item.get('chain_id', 'unknown')}")
    logger.info(f"  多链?: {first_item.get('if_multi_chain', False)}")
    
    # 检查RNA字典
    rna_dic = first_item.get('rna_dic', {})
    logger.info(f"  RNA字典包含 {len(rna_dic)} 个残基")
    
    if len(rna_dic) > 0:
        first_res_id = list(rna_dic.keys())[0]
        first_res = rna_dic[first_res_id]
        logger.info(f"  第一个残基 (ID={first_res_id}):")
        logger.info(f"    残基名称: {first_res.get('residue_name', 'unknown')}")
        atom_coords = first_res.get('atom_coords', {})
        logger.info(f"    原子数量: {len(atom_coords)}")
        logger.info(f"    原子类型: {list(atom_coords.keys())[:5]}...")
    
    # 使用适配器转换数据
    logger.info("开始适配转换...")
    adapted_data = adapt_training_dict_single(data)
    
    # 检查结果
    logger.info(f"适配结果: {len(adapted_data)} 个结构")
    
    if len(adapted_data) > 0:
        first_adapted = adapted_data[0]
        logger.info(f"第一个适配后的结构:")
        logger.info(f"  PDB ID: {first_adapted.get('pdb_id', 'unknown')}")
        logger.info(f"  序列长度: {len(first_adapted.get('sequence', ''))}")
        logger.info(f"  序列: {first_adapted.get('sequence', '')[:20]}...")
        
        # 检查扭转角
        torsion_angles = first_adapted.get('torsion_angles', {})
        torsion_masks = first_adapted.get('torsion_masks', {})
        
        for angle_name in torsion_angles:
            angles = torsion_angles[angle_name]
            masks = torsion_masks[angle_name]
            valid_count = sum(masks)
            
            logger.info(f"  {angle_name}: 总计 {len(angles)} 个值, 有效 {valid_count} 个")
            
            # 输出一些有效值示例
            valid_examples = [angles[i] for i, mask in enumerate(masks) if mask > 0][:3]
            if valid_examples:
                logger.info(f"    示例值: {valid_examples}")
    
    # 保存适配后的数据
    if output_dir and len(adapted_data) > 0:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "adapted_data.pt")
        torch.save(adapted_data, output_file)
        logger.info(f"适配后的数据已保存到: {output_file}")
    
    return adapted_data

if __name__ == "__main__":
    import sys
    
    pkl_file = "dataset/Training_Dict_single.pkl"
    output_dir = "dataset/adapted"
    
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    test_adapter(pkl_file, output_dir)