package linear

import (
	"context"
	"testing"

	"github.com/erni27/regression/internal/regressiontest"
	"github.com/erni27/regression/options"
)

func TestRun_WithGradientDescent(t *testing.T) {
	type expected struct {
		r2     float64
		coeffs []float64
	}
	tests := []struct {
		name    string
		path    string
		options options.Options
		want    expected
	}{
		{
			name:    "batch iterative n=1 m=97 alpha=0.0001 i=1500",
			path:    "n=1_m=97.txt",
			options: options.WithIterativeConvergance(0.0001, options.Batch, options.None, 2000),
			want:    expected{r2: 0.702, coeffs: []float64{-3.776, 1.181}},
		},
		{
			name:    "stochastic iterative n=1 m=97 alpha=0.0001 i=150000",
			path:    "n=1_m=97.txt",
			options: options.WithIterativeConvergance(0.0001, options.Stochastic, options.None, 150000),
			want:    expected{r2: 0.7, coeffs: []float64{-3.583, 1.187}},
		},
		{
			name:    "batch automatic n=1 m=97 alpha=0.0001 t=0.0000001",
			path:    "n=1_m=97.txt",
			options: options.WithAutomaticConvergance(0.0001, options.Batch, options.None, 0.0000001),
			want:    expected{r2: 0.702, coeffs: []float64{-3.858, 1.189}},
		},
		{
			name:    "stochastic automatic n=1 m=97 alpha=0.0001 t=0.0000000001",
			path:    "n=1_m=97.txt",
			options: options.WithAutomaticConvergance(0.0001, options.Stochastic, options.None, 0.0000000001),
			want:    expected{r2: 0.702, coeffs: []float64{-3.126, 1.116}},
		},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := WithGradientDescent(tt.options)
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
			if !regressiontest.AreFloatEqual(r2, tt.want.r2, 2) {
				t.Errorf("got r2 %v, want %v", r2, tt.want.r2)
			}
		})
	}
}
