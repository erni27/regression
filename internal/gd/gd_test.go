package gd

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/regressiontest"
	"github.com/erni27/regression/options"
)

func hyphothesis(x []float64, coeffs []float64) (float64, error) {
	if len(x) != len(coeffs) {
		return 0, errors.New("invalid arguments")
	}
	var r float64
	for i := 0; i < len(x); i++ {
		r += x[i] * coeffs[i]
	}
	return r, nil
}

func cost(x [][]float64, y []float64, coeffs []float64) (float64, error) {
	m := len(x)
	var c float64
	for i := 0; i < m; i++ {
		hr, err := hyphothesis(x[i], coeffs)
		if err != nil {
			return 0, err
		}
		c += math.Pow(hr-y[i], 2)
	}
	return c / float64((2 * m)), nil
}

var gradientDescent GradientDescent = New(hyphothesis, cost)

func TestRun(t *testing.T) {
	x := [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
	}
	y := []float64{3, 7, 11}
	tests := []struct {
		name string
		opt  options.Options
		want []float64
	}{
		{
			name: "batch iterative alpha=0.01 i=10",
			opt:  options.WithIterativeConvergance(0.01, options.Batch, 10),
			want: []float64{0.874, 1.1},
		},
		{
			name: "batch automatic alpha=0.01 t=0.01",
			opt:  options.WithAutomaticConvergance(0.01, options.Batch, 0.01),
			want: []float64{0.872, 1.101},
		},
		{
			name: "stochastic iterative alpha=0.01 i=10000",
			opt:  options.WithIterativeConvergance(0.01, options.Stochastic, 10000),
			want: []float64{1, 1},
		},
		{
			name: "stochastic automatic alpha=0.01 t=0.01",
			opt:  options.WithAutomaticConvergance(0.01, options.Stochastic, 0.01),
			want: []float64{0.867, 1.105},
		},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := gradientDescent.Run(ctx, tt.opt, x, y)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !regressiontest.AreFloatSlicesEqual(got, tt.want, 3) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRun_TooLargeLearningRate(t *testing.T) {
	x := [][]float64{
		{100, 200},
		{300, 400},
		{550, 6660},
	}
	y := []float64{333, 777, 1212}
	tests := []struct {
		name string
		opt  options.Options
	}{
		{
			name: "batch iterative alpha=0.01 i=100",
			opt:  options.WithIterativeConvergance(0.01, options.Batch, 100),
		},
		{
			name: "batch automatic alpha=0.6 t=0.001",
			opt:  options.WithAutomaticConvergance(0.6, options.Batch, 0.001),
		},
		{
			name: "stochastic iterative alpha=0.2 i=100000",
			opt:  options.WithIterativeConvergance(0.2, options.Stochastic, 100000),
		},
		{
			name: "stochastic automatic alpha=0.1 t=0.01",
			opt:  options.WithAutomaticConvergance(0.1, options.Stochastic, 0.01),
		},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := gradientDescent.Run(ctx, tt.opt, x, y)
			fmt.Print(got)
			if err != regression.ErrCannotConverge {
				t.Fatalf("want %v, got %v", regression.ErrCannotConverge, err)
			}
		})
	}
}
