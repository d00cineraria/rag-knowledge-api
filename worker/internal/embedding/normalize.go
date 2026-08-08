package embedding

import "math"

// L2Normalize はベクトルをL2ノルムで正規化した新しいスライスを返す。
// ノルムが0（ゼロベクトル）の場合は入力をそのまま返す。
// Python側 app/services/ingest/embedding.py の _l2_normalize と同一仕様。
func L2Normalize(values []float64) []float64 {
	var sumSq float64
	for _, v := range values {
		sumSq += v * v
	}
	norm := math.Sqrt(sumSq)
	if norm == 0 {
		return values
	}

	out := make([]float64, len(values))
	for i, v := range values {
		out[i] = v / norm
	}
	return out
}
