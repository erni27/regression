package matrix

import (
	"math"
	"testing"
)

func TestInverse(t *testing.T) {
	tests := []struct {
		name   string
		matrix [][]float64
		want   [][]float64
	}{
		{
			name: "4x4 without zeros on main diagonal",
			matrix: [][]float64{
				{3, 3, -4, -3},
				{0, 6, 1, 1},
				{5, 4, 2, 1},
				{2, 3, 3, 2},
			},
			want: [][]float64{
				{-7, 5, 12, -19},
				{3, -2, -5, 8},
				{41, -30, -69, 111},
				{-59, 43, 99, -159},
			},
		},
		{
			name: "4x4 with zeros on main diagonal",
			matrix: [][]float64{
				{0, 1, -1, 1},
				{2, 2, 0, -2},
				{1, 1, -2, 0},
				{0, 1, 2, 0},
			},
			want: [][]float64{
				{-4, -2, 5, 3},
				{2, 1, -2, -1},
				{-1, -0.5, 1, 1},
				{-2, -1.5, 3, 2},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Inverse(tt.matrix)
			if err != nil {
				t.Fatalf("Inverse(m [][]float64) error = %v", err)
				return
			}
			n := len(tt.matrix)
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					if round(got[i][j], 6) != tt.want[i][j] {
						t.Fatalf("Inverse(m [][]float64) = %v, want %v", got, tt.want)
					}
				}
			}
		})
	}
}

func round(v float64, p int) float64 {
	r := math.Pow(10, float64(p))
	return math.Round(v*r) / r
}
