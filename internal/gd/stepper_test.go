package gd

import (
	"testing"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/regressiontest"
	"github.com/erni27/regression/options"
)

func TestNewStepper(t *testing.T) {
	tests := []struct {
		name    string
		gdv     options.GradientDescentVariant
		wantErr bool
		err     error
	}{
		{name: "batch", gdv: options.Batch, wantErr: false},
		{name: "stochastic", gdv: options.Stochastic, wantErr: false},
		{name: "unsupported", gdv: 0, wantErr: true, err: regression.ErrUnsupportedGradientDescentVariant},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewStepper(tt.gdv, nil, [][]float64{{1}}, nil, 0)
			if (err != nil) != tt.wantErr {
				if err != tt.err {
					t.Fatalf("want error %v, got %v", tt.err, err)
				}
			}
			if !tt.wantErr && got == nil {
				t.Fatalf("want not nil stepper")
			}
		})
	}
}

func TestTakeStep(t *testing.T) {
	tests := []struct {
		name string
		gdv  options.GradientDescentVariant
		x    [][]float64
		y    []float64
		lr   float64
		want []float64
	}{
		{
			name: "batch n=1 m=10 alpha=0.1",
			gdv:  options.Batch,
			x: [][]float64{
				{1, 4},
				{1, 10},
				{1, 21},
				{1, 22},
				{1, 29},
				{1, 35},
				{1, 39},
				{1, 44},
				{1, 50},
				{1, 51},
			},
			y:    []float64{50, 33, 14, 13.5, 10, 8, 7.6, 7, 5, 4},
			lr:   0.1,
			want: []float64{15.21, 274.94},
		},
		{
			name: "batch n=2 m=5 alpha=0.01",
			gdv:  options.Batch,
			x: [][]float64{
				{1, 5, 7},
				{1, 13, 3},
				{1, 21, 1},
				{1, 6, 6},
				{1, 1, 14},
			},
			y:    []float64{12.5, 15, 25, 10, 13},
			lr:   0.01,
			want: []float64{0.755, 8.555, 3.995},
		},
		{
			name: "stochastic n=1 m=7 alpha=0.01",
			gdv:  options.Stochastic,
			x: [][]float64{
				{1, 9},
				{1, 21},
				{1, 22},
				{1, 39},
				{1, 44},
				{1, 50},
				{1, 51},
			},
			y:    []float64{33, 14, 13.5, 7.6, 7, 5, 4},
			lr:   0.01,
			want: []float64{0.33, 2.97},
		},
		{
			name: "stochastic n=2 m=6 alpha=0.1",
			gdv:  options.Stochastic,
			x: [][]float64{
				{1, 5, 7},
				{1, 13, 3},
				{1, 21, 1},
				{1, 6, 6},
				{1, 1, 14},
				{1, 3, 9},
			},
			y:    []float64{12.5, 15, 25, 10, 13, 9},
			lr:   0.1,
			want: []float64{1.25, 6.25, 8.75},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewStepper(tt.gdv, hyphoStub, tt.x, tt.y, tt.lr)
			if err != nil {
				t.Fatal(err)
			}
			err = s.TakeStep()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if got := s.CurrentCoefficients(); !regressiontest.AreFloatSlicesEqual(got, tt.want, 3) {
				t.Fatalf("want %v, got %v", tt.want, got)
			}
		})
	}
}
