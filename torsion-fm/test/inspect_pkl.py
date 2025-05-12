# inspect_pkl.py
import pickle
import sys

def inspect_pkl(pkl_path):
    """详细检查PKL文件的结构"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"数据类型: {type(data)}")
    
    if isinstance(data, dict):
        print(f"字典键数量: {len(data.keys())}")
        print(f"部分键: {list(data.keys())[:10]}")
        
        # 检查第一个元素的结构
        first_key = list(data.keys())[0]
        first_item = data[first_key]
        print(f"\n第一个元素 (键={first_key}) 的类型: {type(first_item)}")
        
        if isinstance(first_item, dict):
            print(f"第一个元素的键: {list(first_item.keys())}")
            
            # 如果有rna_dic键，检查其结构
            if 'rna_dic' in first_item:
                print(f"\nrna_dic类型: {type(first_item['rna_dic'])}")
                print(f"rna_dic键: {list(first_item['rna_dic'].keys())[:5] if len(first_item['rna_dic']) > 5 else list(first_item['rna_dic'].keys())}")
                
                # 检查第一个残基
                if len(first_item['rna_dic']) > 0:
                    first_residue_key = list(first_item['rna_dic'].keys())[0]
                    first_residue = first_item['rna_dic'][first_residue_key]
                    print(f"\n第一个残基 (键={first_residue_key}): {first_residue.keys() if isinstance(first_residue, dict) else type(first_residue)}")
            
            # 如果没有rna_dic键但有atom_coords键，检查其结构
            elif 'atom_coords' in first_item:
                print(f"\natom_coords类型: {type(first_item['atom_coords'])}")
                print(f"atom_coords键: {list(first_item['atom_coords'].keys())[:5] if isinstance(first_item['atom_coords'], dict) and len(first_item['atom_coords']) > 5 else first_item['atom_coords']}")
            
            # 尝试找到可能包含原子坐标的键
            for key in first_item:
                if isinstance(first_item[key], dict) and len(first_item[key]) > 0:
                    print(f"\n可能的原子坐标键 '{key}': {list(first_item[key].keys())[:5]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python inspect_pkl.py <pkl文件路径>")
        sys.exit(1)
    
    inspect_pkl(sys.argv[1])