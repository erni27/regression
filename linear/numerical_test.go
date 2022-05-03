package linear

import (
	"testing"

	"github.com/erni27/regression/internal/regressiontest"
	"github.com/erni27/regression/opt"
)

func TestRunNumerical(t *testing.T) {
	type expected struct {
		r2     float64
		coeffs []float64
	}
	tests := []struct {
		name     string
		fileName string
		options  opt.Options
		want     expected
	}{
		{
			name:     "batch_iterative_n=1_m=97_alpha=0.01_i=1500",
			fileName: "n=1_m=97.txt",
			options:  opt.WithIterativeConvergance(0.01, opt.Batch, 1500),
			want:     expected{r2: 0.702, coeffs: []float64{-3.630, 1.166}},
		},
		{
			name:     "stochastic_iterative_n=1_m=97_alpha=0.01_i=150000",
			fileName: "n=1_m=97.txt",
			options:  opt.WithIterativeConvergance(0.01, opt.Stochastic, 150000),
			want:     expected{r2: 0.686, coeffs: []float64{-3.653, 1.092}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := WithGradientDescent(tt.options)
			ts, err := regressiontest.LoadTrainingSet(tt.fileName)
			if err != nil {
				t.Fatalf("cannot load training set %v", err)
			}
			got, err := r.Run(ts)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			coeffs, err := got.Coefficients()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !regressiontest.AreFloatSlicesEqual(coeffs, tt.want.coeffs, 3) {
				t.Errorf("got coefficients %v, want %v", coeffs, tt.want.coeffs)
			}
			r2, err := got.Accuracy()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !regressiontest.AreFloatEqual(r2, tt.want.r2, 2) {
				t.Errorf("got r2 %v, want %v", r2, tt.want.r2)
			}
		})
	}
}
