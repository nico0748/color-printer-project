import numpy as np
import re
from typing import Dict, Tuple, Any

WEIGHT_THRESHOLD = 1.5e-3    # 重み用閾値（推奨: 1e-3）
BIAS_THRESHOLD = 1e-6     # バイアス用閾値（推奨: 1e-4）

class ModelParameterProcessor:
    def __init__(self, weight_threshold: float = WEIGHT_THRESHOLD, bias_threshold: float = BIAS_THRESHOLD):
        self.weight_threshold = weight_threshold
        self.bias_threshold = bias_threshold
        self.arrays = {}
    
    @property 
    def threshold(self):
        """後方互換性のため"""
        return self.weight_threshold
        
    def get_threshold_for_param(self, param_name: str) -> float:
        """パラメータタイプに応じた閾値を返す"""
        if 'bias' in param_name:
            return self.bias_threshold
        else:
            return self.weight_threshold
    
    def parse_model_parameters(self, file_path: str) -> Dict[str, np.ndarray]:
        """model_parameters.hファイルを解析して重みとバイアスを抽出"""
        print(f"📖 {file_path}を解析中...")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # float配列のパターンにマッチ
        pattern = r'float\s+(\w+)\[\]\s*=\s*\{([^}]+)\};'
        matches = re.findall(pattern, content)
        
        arrays = {}
        for name, values_str in matches:
            # 数値を抽出して配列に変換
            numbers = re.findall(r'-?[\d.]+(?:[eE][-+]?\d+)?', values_str)
            values = [float(num_str) for num_str in numbers]
            arrays[name] = np.array(values, dtype=np.float32)
            print(f"  ✓ {name}: {len(values)}個の要素")
        
        self.arrays = arrays
        return arrays
    
    def detect_network_structure(self) -> list:
        """実際の配列サイズからネットワーク構造を自動検出"""
        print("\n🔍 ネットワーク構造を自動検出中...")
        
        # 各配列のサイズを確認
        sizes = {name: len(array) for name, array in self.arrays.items()}
        print(f"   配列サイズ: {sizes}")
        
        # バイアスサイズからレイヤーサイズを推定
        bias_sizes = {name: size for name, size in sizes.items() if 'bias' in name}
        print(f"   バイアスサイズ: {bias_sizes}")
        
        # weight配列のサイズから構造を推定
        network_structure = []
        
        # Layer 1の推定
        if 'weight_1' in sizes and 'bias_1' in sizes:
            weight1_size = sizes['weight_1']
            bias1_size = sizes['bias_1']
            input_size = 3  # RGB入力
            output_size = bias1_size
            
            if weight1_size == input_size * output_size:
                network_structure.append(('weight_1', input_size, output_size))
                network_structure.append(('bias_1', output_size, 1))
                print(f"   Layer 1: 入力{input_size} → 隠れ層{output_size}")
            else:
                print(f"   ⚠️ Layer 1のサイズが一致しません: {weight1_size} ≠ {input_size}×{output_size}")
        
        # Layer 2の推定
        if 'weight_2' in sizes and 'bias_2' in sizes:
            weight2_size = sizes['weight_2']
            bias2_size = sizes['bias_2']
            input_size = bias1_size  # 前層の出力サイズ
            output_size = bias2_size
            
            if weight2_size == input_size * output_size:
                network_structure.append(('weight_2', input_size, output_size))
                network_structure.append(('bias_2', output_size, 1))
                print(f"   Layer 2: 隠れ層{input_size} → 隠れ層{output_size}")
            else:
                print(f"   ⚠️ Layer 2のサイズが一致しません: {weight2_size} ≠ {input_size}×{output_size}")
        
        # Layer 3の推定
        if 'weight_3' in sizes and 'bias_3' in sizes:
            weight3_size = sizes['weight_3']
            bias3_size = sizes['bias_3']
            input_size = bias2_size  # 前層の出力サイズ
            output_size = bias3_size
            
            if weight3_size == input_size * output_size:
                network_structure.append(('weight_3', input_size, output_size))
                network_structure.append(('bias_3', output_size, 1))
                print(f"   Layer 3: 隠れ層{input_size} → 出力{output_size}")
            else:
                print(f"   ⚠️ Layer 3のサイズが一致しません: {weight3_size} ≠ {input_size}×{output_size}")
        
        print(f"   検出された構造: {network_structure}")
        return network_structure
    
    def prune_weights(self, weights: np.ndarray, name: str = "unknown") -> Tuple[np.ndarray, float]:
        """閾値以下の重みを0にプルーニング（詳細デバッグ付き）"""
        # パラメータタイプに応じた閾値を取得
        threshold = self.get_threshold_for_param(name)
        
        print(f"\n🔧 {name} のプルーニング処理:")
        print(f"   使用閾値: {threshold} ({'バイアス用' if 'bias' in name else '重み用'})")
        
        # 元の統計
        original_weights = weights.copy()
        original_nonzero = np.count_nonzero(weights)
        original_abs_below_threshold = np.sum(np.abs(weights) < threshold)
        original_abs_above_threshold = np.sum(np.abs(weights) >= threshold)
        
        print(f"   元の非ゼロ要素: {original_nonzero}/{len(weights)} ({original_nonzero/len(weights)*100:.1f}%)")
        print(f"   元の閾値以下要素: {original_abs_below_threshold} ({original_abs_below_threshold/len(weights)*100:.1f}%)")
        print(f"   元の閾値以上要素: {original_abs_above_threshold} ({original_abs_above_threshold/len(weights)*100:.1f}%)")
        
        # 分布分析（閾値を渡す）
        distribution = self.analyze_weight_distribution(weights, name, threshold)
        
        # プルーニング実行
        pruned_weights = weights.copy()
        mask = np.abs(pruned_weights) < threshold
        pruned_count = np.sum(mask)
        pruned_weights[mask] = 0
        
        # プルーニング後の統計
        final_nonzero = np.count_nonzero(pruned_weights)
        
        print(f"\n   プルーニング実行結果:")
        print(f"   ゼロにした要素数: {pruned_count}")
        print(f"   プルーニング後非ゼロ要素: {final_nonzero}/{len(weights)} ({final_nonzero/len(weights)*100:.1f}%)")
        
        # プルーニング率の計算
        pruning_ratio = pruned_count / len(weights) * 100
        compression_ratio = (original_nonzero - final_nonzero) / len(weights) * 100
        
        print(f"   プルーニング率: {pruning_ratio:.2f}% (ゼロにした要素/全要素)")
        print(f"   圧縮率: {compression_ratio:.2f}% (減少した非ゼロ要素/全要素)")
        
        # プルーニングされた値の分析
        if pruned_count > 0:
            pruned_values = original_weights[mask]
            pruned_abs_values = np.abs(pruned_values)
            print(f"   プルーニングされた値の範囲: {np.min(pruned_abs_values):.8f} ~ {np.max(pruned_abs_values):.8f}")
            print(f"   プルーニングされた値の平均: {np.mean(pruned_abs_values):.8f}")
        else:
            print(f"   ⚠️ プルーニングされた要素がありません！")
            print(f"   → 全ての要素が閾値({threshold})以上です")
        
        return pruned_weights, pruning_ratio
    
    def analyze_weight_distribution(self, weights: np.ndarray, name: str, threshold: float = None) -> Dict[str, Any]:
        """重みの分布を分析"""
        if threshold is None:
            threshold = self.get_threshold_for_param(name)
            
        abs_weights = np.abs(weights)
        
        stats = {
            'min': float(np.min(abs_weights)),
            'max': float(np.max(abs_weights)),
            'mean': float(np.mean(abs_weights)),
            'median': float(np.median(abs_weights)),
            'std': float(np.std(abs_weights)),
            'below_threshold': int(np.sum(abs_weights < threshold)),
            'above_threshold': int(np.sum(abs_weights >= threshold)),
            'zeros': int(np.sum(abs_weights == 0)),
            'total_elements': len(weights)
        }
        
        # パーセンタイル情報
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}'] = float(np.percentile(abs_weights, p))
        
        print(f"\n🔍 {name} の重み分布分析:")
        print(f"   要素数: {stats['total_elements']}")
        print(f"   ゼロ要素: {stats['zeros']} ({stats['zeros']/stats['total_elements']*100:.1f}%)")
        print(f"   閾値({threshold})以下: {stats['below_threshold']} ({stats['below_threshold']/stats['total_elements']*100:.1f}%)")
        print(f"   閾値以上: {stats['above_threshold']} ({stats['above_threshold']/stats['total_elements']*100:.1f}%)")
        print(f"   最小値: {stats['min']:.8f}")
        print(f"   最大値: {stats['max']:.8f}")
        print(f"   平均値: {stats['mean']:.8f}")
        print(f"   中央値: {stats['median']:.8f}")
        print(f"   標準偏差: {stats['std']:.8f}")
        
        # 閾値周辺の値をサンプル表示
        near_threshold = abs_weights[(abs_weights > threshold/10) & (abs_weights < threshold*10)]
        if len(near_threshold) > 0:
            print(f"   閾値周辺の値サンプル: {sorted(near_threshold)[:10]}")
        
        return stats
    
    def dense_to_csr(self, matrix: np.ndarray) -> Dict[str, Any]:
        """密行列をCSR形式に変換"""
        if len(matrix.shape) == 1:
            # 1次元配列（バイアス）の場合
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
            # 2次元配列（重み）の場合
            rows, cols = np.nonzero(matrix)
            values = matrix[rows, cols]
            
            # 行ポインタを作成
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
        expected_size = input_size * output_size
        actual_size = len(weight_array)
        
        if actual_size != expected_size:
            raise ValueError(f"配列サイズが一致しません: 実際={actual_size}, 期待={expected_size} ({input_size}×{output_size})")
        
        return weight_array.reshape(output_size, input_size)
    
    def process_all_layers(self) -> Dict[str, Dict[str, Any]]:
        """全レイヤーの処理を実行"""
        # ネットワーク構造を自動検出
        network_structure = self.detect_network_structure()
        
        if not network_structure:
            raise ValueError("ネットワーク構造を検出できませんでした")
        
        processed_layers = {}
        total_original = 0
        total_compressed = 0
        
        print(f"\n🔧 重み用閾値: {self.weight_threshold}")
        print(f"🔧 バイアス用閾値: {self.bias_threshold}")
        print(f"🔧 重み閾値の科学記法: {self.weight_threshold:.2e}")
        print(f"🔧 バイアス閾値の科学記法: {self.bias_threshold:.2e}")
        print("=" * 80)
        
        for param_name, input_size, output_size in network_structure:
            if param_name not in self.arrays:
                print(f"⚠️  {param_name}が見つかりません")
                continue
            
            original_array = self.arrays[param_name]
            total_original += len(original_array)
            
            print(f"\n📊 {param_name} の処理開始:")
            print(f"   配列サイズ: {len(original_array)}")
            print(f"   形状情報: 入力{input_size} → 出力{output_size}")
            
            # プルーニング実行（デバッグ情報付き）
            pruned_array, pruning_ratio = self.prune_weights(original_array, param_name)
            
            # バイアスは1次元のまま、重みは2次元に変形
            if 'weight' in param_name:
                matrix = self.reshape_weight_to_matrix(pruned_array, input_size, output_size)
            else:
                matrix = pruned_array
            
            # CSR形式に変換
            csr_data = self.dense_to_csr(matrix)
            
            # 圧縮率計算
            compression_ratio = (1 - csr_data['nnz'] / len(original_array)) * 100
            total_compressed += csr_data['nnz']
            
            # 結果を保存
            processed_layers[param_name] = {
                'original_size': len(original_array),
                'csr_data': csr_data,
                'pruning_ratio': pruning_ratio,
                'compression_ratio': compression_ratio,
                'input_size': input_size,
                'output_size': output_size
            }
            
            print(f"\n📈 {param_name} の最終結果:")
            print(f"   元サイズ: {len(original_array)} → 非ゼロ: {csr_data['nnz']}")
            print(f"   プルーニング率: {pruning_ratio:.2f}%")
            print(f"   CSR圧縮率: {compression_ratio:.2f}%")
        
        print("\n" + "=" * 80)
        overall_compression = (1 - total_compressed / total_original) * 100
        print(f"🎯 全体統計:")
        print(f"   元パラメータ数: {total_original}")
        print(f"   圧縮後パラメータ数: {total_compressed}")
        print(f"   全体圧縮率: {overall_compression:.2f}%")
        print(f"   推定メモリ削減: {(total_original - total_compressed) * 4}バイト")
        
        return processed_layers
    
    def generate_csr_header(self, processed_layers: Dict[str, Dict[str, Any]]) -> str:
        """CSR形式のヘッダファイルを生成"""
        print(f"\n📝 CSRヘッダファイルを生成中...")
        
        header_content = f"""// CSR形式で圧縮されたニューラルネットワークパラメータ
// 重み用プルーニング閾値: {self.weight_threshold}
// バイアス用プルーニング閾値: {self.bias_threshold}
// 自動生成ファイル - 手動編集禁止
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
            
            header_content += f"""
// {param_name} - CSR形式
// 元サイズ: {layer_info['original_size']}, 非ゼロ要素: {csr['nnz']}, 圧縮率: {layer_info['compression_ratio']:.1f}%
const float {param_name}_values[] CSR_PROGMEM = {{"""
            
            # 値の配列
            for i, val in enumerate(csr['values']):
                if i % 8 == 0:
                    header_content += "\n    "
                header_content += f"{val:.8f}f"
                if i < len(csr['values']) - 1:
                    header_content += ", "
            header_content += "\n};\n"
            
            # インデックス配列
            header_content += f"""
const int32_t {param_name}_indices[] CSR_PROGMEM = {{"""
            for i, idx in enumerate(csr['indices']):
                if i % 12 == 0:
                    header_content += "\n    "
                header_content += f"{idx}"
                if i < len(csr['indices']) - 1:
                    header_content += ", "
            header_content += "\n};\n"
            
            # 行ポインタ配列（2次元の場合のみ）
            if not csr['is_1d']:
                header_content += f"""
const int32_t {param_name}_indptr[] CSR_PROGMEM = {{"""
                for i, ptr in enumerate(csr['indptr']):
                    if i % 12 == 0:
                        header_content += "\n    "
                    header_content += f"{ptr}"
                    if i < len(csr['indptr']) - 1:
                        header_content += ", "
                header_content += "\n};\n"

        # 構造体定義を追加
        header_content += """
// CSR形式のデータ構造
typedef struct {
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

        # 各パラメータの構造体インスタンスを生成
        for param_name, layer_info in processed_layers.items():
            csr = layer_info['csr_data']
            
            if csr['is_1d']:
                header_content += f"""
const csr_vector_t {param_name}_csr = {{
    {param_name}_values,
    {param_name}_indices,
    {csr['nnz']},
    {csr['shape'][0]}
}};
"""
            else:
                header_content += f"""
const csr_matrix_t {param_name}_csr = {{
    {param_name}_values,
    {param_name}_indices,
    {param_name}_indptr,
    {csr['nnz']},
    {csr['shape'][0]},
    {csr['shape'][1]}
}};
"""

        # ヘルパー関数の追加
        header_content += """
// CSR形式から値を取得するヘルパー関数
#ifdef ARDUINO
#define READ_PROGMEM_FLOAT(addr) pgm_read_float(addr)
#define READ_PROGMEM_DWORD(addr) pgm_read_dword(addr)
#else
#define READ_PROGMEM_FLOAT(addr) (*(addr))
#define READ_PROGMEM_DWORD(addr) (*(addr))
#endif

// CSR行列から値を取得
static inline float get_csr_weight(const csr_matrix_t* csr, int row, int col) {
    int32_t start = READ_PROGMEM_DWORD(&csr->indptr[row]);
    int32_t end = READ_PROGMEM_DWORD(&csr->indptr[row + 1]);
    
    for (int32_t i = start; i < end; i++) {
        if (READ_PROGMEM_DWORD(&csr->indices[i]) == col) {
            return READ_PROGMEM_FLOAT(&csr->values[i]);
        }
    }
    return 0.0f;  // スパース要素は0
}

// CSRベクトルから値を取得
static inline float get_csr_bias(const csr_vector_t* csr, int index) {
    for (int32_t i = 0; i < csr->nnz; i++) {
        if (READ_PROGMEM_DWORD(&csr->indices[i]) == index) {
            return READ_PROGMEM_FLOAT(&csr->values[i]);
        }
    }
    return 0.0f;  // スパース要素は0
}

#endif // MODEL_PARAMETERS_CSR_H
"""
        
        return header_content

def main():
    """メイン処理"""
    print("🚀 model_parameters.h → model_parameters_csr.h 変換処理を開始します...")
    print(f"⚙️ 設定された閾値: 重み={WEIGHT_THRESHOLD:.0e}, バイアス={BIAS_THRESHOLD:.0e}")
    
    # 重みとバイアスで異なる閾値テスト
    test_configs = [
        {"weight": 1e-4, "bias": 1e-5, "description": "重み1e-4, バイアス1e-5"},
        {"weight": 1e-3, "bias": 1e-4, "description": "重み1e-3, バイアス1e-4"},
        {"weight": 1e-2, "bias": 1e-3, "description": "重み1e-2, バイアス1e-3"},
        {"weight": 1e-1, "bias": 1e-2, "description": "重み1e-1, バイアス1e-2"},
    ]
    
    print(f"\n🧪 異なる閾値での圧縮率テスト:")
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"🔍 {config['description']}:")
        print(f"{'='*60}")
        
        processor = ModelParameterProcessor(
            weight_threshold=config["weight"], 
            bias_threshold=config["bias"]
        )
        
        try:
            arrays = processor.parse_model_parameters("model_parameters.h")
            
            if not arrays:
                print("❌ 配列が見つかりませんでした")
                continue
            
            # 簡易統計表示
            total_prunable = 0
            for name, array in arrays.items():
                threshold = processor.get_threshold_for_param(name)
                abs_array = np.abs(array)
                below_threshold = np.sum(abs_array < threshold)
                total_prunable += below_threshold
                print(f"   {name}: {below_threshold}/{len(array)}要素が閾値({threshold:.0e})以下 ({below_threshold/len(array)*100:.1f}%)")
            
            total_elements = sum(len(array) for array in arrays.values())
            print(f"   📊 全体: {total_prunable}/{total_elements}要素がプルーニング可能 ({total_prunable/total_elements*100:.1f}%)")
            
        except FileNotFoundError:
            print("❌ エラー: model_parameters.hファイルが見つかりません")
            return
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            return
    
    # 実際の変換処理（設定された閾値を使用）
    print(f"\n{'='*60}")
    print(f"🚀 実際の変換処理 (重み: {WEIGHT_THRESHOLD:.0e}, バイアス: {BIAS_THRESHOLD:.0e}):")
    print(f"{'='*60}")
    
    processor = ModelParameterProcessor(weight_threshold=WEIGHT_THRESHOLD, bias_threshold=BIAS_THRESHOLD)
    
    try:
        # model_parameters.hを解析
        processor.parse_model_parameters("model_parameters.h")
        
        # 全レイヤーを処理
        processed_layers = processor.process_all_layers()
        
        # CSRヘッダファイルを生成
        header_content = processor.generate_csr_header(processed_layers)
        with open("model_parameters_csr.h", "w") as f:
            f.write(header_content)
        
        print(f"\n✅ 変換完了！")
        print(f"   📄 model_parameters_csr.h が生成されました")
        print(f"   🎯 全体圧縮率: {(1 - sum(layer['csr_data']['nnz'] for layer in processed_layers.values()) / sum(layer['original_size'] for layer in processed_layers.values())) * 100:.1f}%")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()