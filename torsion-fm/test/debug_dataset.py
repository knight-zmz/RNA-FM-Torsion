# debug_dataset.py
import os
import glob
import logging
import pickle
from data.preprocessing import process_pdb_file, load_pdb_data

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_dataset(data_dir):
    """调试数据集加载问题"""
    logger.info(f"正在调试数据集目录: {data_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return
        
    # 查找所有pkl文件
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    logger.info(f"找到 {len(pkl_files)} 个PKL文件")
    
    if len(pkl_files) == 0:
        logger.error(f"在目录 {data_dir} 中未找到PKL文件")
        return
    
    # 检查文件内容
    multi_chain_count = 0
    empty_seq_count = 0
    successful_count = 0
    
    for i, pkl_file in enumerate(pkl_files):
        if i < 3:  # 只打印前3个文件的详细信息
            logger.info(f"详细检查文件 {i+1}/{len(pkl_files)}: {os.path.basename(pkl_file)}")
            
            # 尝试读取PKL文件内容
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"文件结构: {list(data.keys())}")
                
                # 检查是否多链
                if data.get('if_multi_chain', False):
                    logger.info(f"文件包含多链RNA")
                    multi_chain_count += 1
                    continue
                
                # 检查RNA字典
                rna_dic = data.get('rna_dic', {})
                logger.info(f"文件包含 {len(rna_dic)} 个残基")
                
                # 提取序列
                chain_id, sequence, atom_coords_dict, is_multi_chain = load_pdb_data(pkl_file)
                if sequence:
                    logger.info(f"成功提取序列: {sequence[:20]}... (长度: {len(sequence)})")
                else:
                    logger.info(f"无法提取序列")
                    empty_seq_count += 1
                    continue
                    
                # 尝试处理完整文件
                result = process_pdb_file(pkl_file)
                if result is not None:
                    logger.info(f"文件处理成功")
                    successful_count += 1
                else:
                    logger.info(f"文件处理失败")
            except Exception as e:
                logger.error(f"读取文件失败: {str(e)}")
        else:
            # 对其余文件只进行简单处理
            try:
                result = process_pdb_file(pkl_file)
                if result is not None:
                    successful_count += 1
                else:
                    # 检查失败原因
                    chain_id, sequence, atom_coords_dict, is_multi_chain = load_pdb_data(pkl_file)
                    if is_multi_chain:
                        multi_chain_count += 1
                    elif sequence is None:
                        empty_seq_count += 1
            except Exception:
                pass
    
    logger.info(f"处理统计:")
    logger.info(f"总文件数: {len(pkl_files)}")
    logger.info(f"多链RNA (已跳过): {multi_chain_count}")
    logger.info(f"无法提取序列: {empty_seq_count}")
    logger.info(f"成功处理: {successful_count}")
    
    if successful_count == 0:
        logger.error("没有成功处理任何文件，数据集将为空!")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("使用方法: python debug_dataset.py <数据目录路径>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    debug_dataset(data_dir)