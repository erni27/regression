package linear

import (
	"testing"

	"github.com/erni27/regression/internal/regressiontest"
	"github.com/erni27/regression/options"
)

func TestRunNumerical(t *testing.T) {
	type expected struct {
		r2     float64
		coeffs []float64
	}
	tests := []struct {
		name     string
		fileName string
		options  options.Options
		want     expected
	}{
		{
			name:     "batch_iterative_n=1_m=97_alpha=0.01_i=1500",
			fileName: "n=1_m=97.txt",
			options:  options.WithIterativeConvergance(0.01, options.Batch, 1500),
			want:     expected{r2: 0.702, coeffs: []float64{-3.630, 1.166}},
		},
		{
			name:     "stochastic_iterative_n=1_m=97_alpha=0.01_i=150000",
			fileName: "n=1_m=97.txt",
			options:  options.WithIterativeConvergance(0.01, options.Stochastic, 150000),
			want:     expected{r2: 0.686, coeffs: []float64{-3.653, 1.092}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := WithGradientDescent(tt.options)
			s, err := regressiontest.LoadTrainingSet(tt.fileName)
			if err != nil {
				t.Fatalf("cannot load training set %v", err)
			}
			got, err := r.Run(*s)
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
