package embedding

import "testing"

func TestL2Normalize(t *testing.T) {
	got := L2Normalize([]float64{3, 4})
	want := []float64{0.6, 0.8}

	for i := range want {
		if diff := got[i] - want[i]; diff > 1e-9 || diff < -1e-9 {
			t.Fatalf("L2Normalize([3,4])[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	var sumSq float64
	for _, v := range got {
		sumSq += v * v
	}
	if diff := sumSq - 1.0; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("normalized vector norm^2 = %v, want 1.0", sumSq)
	}
}

func TestL2NormalizeZeroVector(t *testing.T) {
	in := []float64{0, 0, 0}
	got := L2Normalize(in)

	for i := range in {
		if got[i] != in[i] {
			t.Fatalf("L2Normalize(zero vector) = %v, want unchanged input %v", got, in)
		}
	}
}
