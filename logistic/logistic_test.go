package logistic

import (
	"context"
	"testing"

	"github.com/erni27/regression/internal/regressiontest"
	"github.com/erni27/regression/options"
)

func TestRun_WithGradientDescent(t *testing.T) {
	type expected struct {
		acc    float64
		coeffs []float64
	}
	tests := []struct {
		name    string
		path    string
		options options.Options
		want    expected
	}{
		{
			name:    "batch gd n=2 m=100 alpha=0.01 i=100",
			path:    "n=2_m=100.txt",
			options: options.WithIterativeConvergance(0.01, options.Batch, 100),
			want:    expected{acc: 0.6, coeffs: []float64{-7.465, 33.217, -4.415}},
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
			acc := got.Accuracy()
			if acc != tt.want.acc {
				t.Errorf("got acc %v, want %v", acc, tt.want.acc)
			}
		})
	}
}
