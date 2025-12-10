import numpy as np
import re
from typing import Dict, Tuple, Any

# ==================== 閾値設定 ====================
WEIGHT_THRESHOLD = 1e-3    # 重み用閾値
BIAS_THRESHOLD = 1e-4      # バイアス用閾値
# ================================================

class ModelParameterProcessor:
    def __init__(self, weight_threshold: float = WEIGHT_THRESHOLD, bias_threshold: float = BIAS_THRESHOLD):
        self.weight_threshold = weight_threshold
        self.bias_threshold = bias_threshold
        self.arrays = {}
        
    def get_threshold_for_param(self, param_name: str) -> float:
        """パラメータタイプに応じた閾値を返す"""
        return self.bias_threshold if 'bias' in param_name else self.weight_threshold
    
    def parse_model_parameters(self, file_path: str) -> Dict[str, np.ndarray]:
        """model_parameters.hファイルを解析"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        pattern = r'float\s+(\w+)\[\]\s*=\s*\{([^}]+)\};'
        matches = re.findall(pattern, content)
        
        arrays = {}
        for name, values_str in matches:
            numbers = re.findall(r'-?[\d.]+(?:[eE][-+]?\d+)?', values_str)
            values = [float(num_str) for num_str in numbers]
            arrays[name] = np.array(values, dtype=np.float32)
        
        self.arrays = arrays
        return arrays
    
    def detect_network_structure(self) -> list:
        """ネットワーク構造を自動検出"""
        sizes = {name: len(array) for name, array in self.arrays.items()}
        bias_sizes = {name: size for name, size in sizes.items() if 'bias' in name}
        
        network_structure = []
        
        # Layer 1
        if 'weight_1' in sizes and 'bias_1' in sizes:
            input_size = 3  # RGB入力
            output_size = bias_sizes['bias_1']
            network_structure.append(('weight_1', input_size, output_size))
            network_structure.append(('bias_1', output_size, 1))
        
        # Layer 2
        if 'weight_2' in sizes and 'bias_2' in sizes:
            input_size = bias_sizes['bias_1']
            output_size = bias_sizes['bias_2']
            network_structure.append(('weight_2', input_size, output_size))
            network_structure.append(('bias_2', output_size, 1))
        
        # Layer 3
        if 'weight_3' in sizes and 'bias_3' in sizes:
            input_size = bias_sizes['bias_2']
            output_size = bias_sizes['bias_3']
            network_structure.append(('weight_3', input_size, output_size))
            network_structure.append(('bias_3', output_size, 1))
        
        return network_structure
    
    def prune_weights(self, weights: np.ndarray, name: str) -> np.ndarray:
        """閾値以下の重みを0にプルーニング"""
        threshold = self.get_threshold_for_param(name)
        pruned_weights = weights.copy()
        pruned_weights[np.abs(pruned_weights) < threshold] = 0
        return pruned_weights
    
    def dense_to_csr(self, matrix: np.ndarray) -> Dict[str, Any]:
        """密行列をCSR形式に変換"""
        if len(matrix.shape) == 1:
            nonzero_indices = np.nonzero(matrix)[0]
            values = matrix[nonzero_indices]
            
            return {
                'values': values.astype(np.float32),
                'indices': nonzero_indices.astype(np.int32),
                'indptr': np.array([0, len(nonzero_indices)], dtype=np.int32),
                'shape': matrix.shape,
                'nnz': len(values),
                'is_1d': True
            }
        else:
            rows, cols = np.nonzero(matrix)
            values = matrix[rows, cols]
            
            indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int32)
            for row in rows:
                indptr[row + 1] += 1
            indptr = np.cumsum(indptr)
            
            return {
                'values': values.astype(np.float32),
                'indices': cols.astype(np.int32),
                'indptr': indptr,
                'shape': matrix.shape,
                'nnz': len(values),
                'is_1d': False
            }
    
    def reshape_weight_to_matrix(self, weight_array: np.ndarray, input_size: int, output_size: int) -> np.ndarray:
        """1次元の重み配列を2次元行列に変形"""
        return weight_array.reshape(output_size, input_size)
    
    def process_all_layers(self) -> Dict[str, Dict[str, Any]]:
        """全レイヤーの処理を実行"""
        network_structure = self.detect_network_structure()
        processed_layers = {}
        
        for param_name, input_size, output_size in network_structure:
            if param_name not in self.arrays:
                continue
            
            original_array = self.arrays[param_name]
            pruned_array = self.prune_weights(original_array, param_name)
            
            if 'weight' in param_name:
                matrix = self.reshape_weight_to_matrix(pruned_array, input_size, output_size)
            else:
                matrix = pruned_array
            
            csr_data = self.dense_to_csr(matrix)
            compression_ratio = (1 - csr_data['nnz'] / len(original_array)) * 100
            
            processed_layers[param_name] = {
                'original_size': len(original_array),
                'csr_data': csr_data,
                'compression_ratio': compression_ratio,
                'input_size': input_size,
                'output_size': output_size
            }
        
        return processed_layers
    
    def generate_csr_header(self, processed_layers: Dict[str, Dict[str, Any]]) -> str:
        """CSR形式のヘッダファイルを生成"""
        header_content = f"""// CSR Neural Network Parameters
// Weight threshold: {self.weight_threshold}, Bias threshold: {self.bias_threshold}
#ifndef MODEL_PARAMETERS_CSR_H
#define MODEL_PARAMETERS_CSR_H

#include <stdint.h>
#ifdef ARDUINO
#include <avr/pgmspace.h>
#define CSR_PROGMEM PROGMEM
#else
#define CSR_PROGMEM
#endif

"""

        # 各レイヤーのデータを生成
        for param_name, layer_info in processed_layers.items():
            csr = layer_info['csr_data']
            
            # values配列
            header_content += f"const float {param_name}_values[] CSR_PROGMEM = {{"
            for i, val in enumerate(csr['values']):
                if i % 8 == 0:
                    header_content += "\n    "
                header_content += f"{val:.8f}f"
                if i < len(csr['values']) - 1:
                    header_content += ", "
            header_content += "\n};\n\n"
            
            # indices配列
            header_content += f"const int32_t {param_name}_indices[] CSR_PROGMEM = {{"
            for i, idx in enumerate(csr['indices']):
                if i % 12 == 0:
                    header_content += "\n    "
                header_content += f"{idx}"
                if i < len(csr['indices']) - 1:
                    header_content += ", "
            header_content += "\n};\n\n"
            
            # indptr配列（2次元の場合のみ）
            if not csr['is_1d']:
                header_content += f"const int32_t {param_name}_indptr[] CSR_PROGMEM = {{"
                for i, ptr in enumerate(csr['indptr']):
                    if i % 12 == 0:
                        header_content += "\n    "
                    header_content += f"{ptr}"
                    if i < len(csr['indptr']) - 1:
                        header_content += ", "
                header_content += "\n};\n\n"

        # 構造体定義
        header_content += """typedef struct {
    const float* values;
    const int32_t* indices;
    const int32_t* indptr;
    int32_t nnz;
    int32_t rows;
    int32_t cols;
} csr_matrix_t;

typedef struct {
    const float* values;
    const int32_t* indices;
    int32_t nnz;
    int32_t size;
} csr_vector_t;

"""

        # 構造体インスタンス
        for param_name, layer_info in processed_layers.items():
            csr = layer_info['csr_data']
            
            if csr['is_1d']:
                header_content += f"""const csr_vector_t {param_name}_csr = {{
    {param_name}_values, {param_name}_indices, {csr['nnz']}, {csr['shape'][0]}
}};

"""
            else:
                header_content += f"""const csr_matrix_t {param_name}_csr = {{
    {param_name}_values, {param_name}_indices, {param_name}_indptr,
    {csr['nnz']}, {csr['shape'][0]}, {csr['shape'][1]}
}};

"""

        # ヘルパー関数
        header_content += """#ifdef ARDUINO
#define READ_PROGMEM_FLOAT(addr) pgm_read_float(addr)
#define READ_PROGMEM_DWORD(addr) pgm_read_dword(addr)
#else
#define READ_PROGMEM_FLOAT(addr) (*(addr))
#define READ_PROGMEM_DWORD(addr) (*(addr))
#endif

static inline float get_csr_weight(const csr_matrix_t* csr, int row, int col) {
    int32_t start = READ_PROGMEM_DWORD(&csr->indptr[row]);
    int32_t end = READ_PROGMEM_DWORD(&csr->indptr[row + 1]);
    for (int32_t i = start; i < end; i++) {
        if (READ_PROGMEM_DWORD(&csr->indices[i]) == col) {
            return READ_PROGMEM_FLOAT(&csr->values[i]);
        }
    }
    return 0.0f;
}

static inline float get_csr_bias(const csr_vector_t* csr, int index) {
    for (int32_t i = 0; i < csr->nnz; i++) {
        if (READ_PROGMEM_DWORD(&csr->indices[i]) == index) {
            return READ_PROGMEM_FLOAT(&csr->values[i]);
        }
    }
    return 0.0f;
}

#endif // MODEL_PARAMETERS_CSR_H
"""
        
        return header_content

def main():
    """メイン処理"""
    processor = ModelParameterProcessor(weight_threshold=WEIGHT_THRESHOLD, bias_threshold=BIAS_THRESHOLD)
    
    try:
        processor.parse_model_parameters("model_parameters.h")
        processed_layers = processor.process_all_layers()
        
        header_content = processor.generate_csr_header(processed_layers)
        with open("model_parameters_csr.h", "w") as f:
            f.write(header_content)
        
        # 簡潔な結果表示
        total_original = sum(layer['original_size'] for layer in processed_layers.values())
        total_compressed = sum(layer['csr_data']['nnz'] for layer in processed_layers.values())
        overall_compression = (1 - total_compressed / total_original) * 100
        
        print(f"Conversion completed successfully!")
        print(f"Original parameters: {total_original}")
        print(f"Compressed parameters: {total_compressed}")
        print(f"Compression ratio: {overall_compression:.1f}%")
        print(f"Output: model_parameters_csr.h")
        
    except FileNotFoundError:
        print("Error: model_parameters.h not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
