package linear

import (
	"context"
	"testing"

	"github.com/erni27/regression/internal/regressiontest"
)

func TestRun_WithNormalEquation(t *testing.T) {
	type expected struct {
		r2     float64
		coeffs []float64
	}
	tests := []struct {
		name string
		path string
		want expected
	}{
		{
			name: "n=1 m=97",
			path: "n=1_m=97.txt",
			want: expected{r2: 0.702, coeffs: []float64{-3.896, 1.193}},
		},
		{
			name: "n=2 m=47",
			path: "n=2_m=47.txt",
			want: expected{r2: 0.733, coeffs: []float64{89597.91, 139.211, -8738.019}},
		},
	}
	r := WithNormalEquation()
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := regressiontest.LoadTrainingSet(tt.path)
			if err != nil {
				t.Fatalf("cannot load training set %v", err)
			}
			got, err := r.Run(ctx, s)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			coeffs := got.Coefficients()
			if !regressiontest.AreFloatSlicesEqual(coeffs, tt.want.coeffs, 3) {
				t.Errorf("got coefficients %v, want %v", coeffs, tt.want.coeffs)
			}
			r2 := got.Accuracy()
			if !regressiontest.AreFloatEqual(r2, tt.want.r2, 3) {
				t.Errorf("got r2 %v, want %v", r2, tt.want.r2)
			}
		})
	}
}
