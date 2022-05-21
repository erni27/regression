package gd

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/erni27/regression/options"
)

type stepperMock struct {
	baseStepper
	takeStep func() error
}

func (s stepperMock) TakeStep() error {
	return s.takeStep()
}

func TestConverge_Iterative(t *testing.T) {
	tests := []struct {
		name string
		i    int
	}{
		{name: "i=1", i: 1},
		{name: "i=10", i: 10},
		{name: "i=100", i: 100},
		{name: "i=237", i: 237},
		{name: "i=1000", i: 1000},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var counter int
			s := stepperMock{takeStep: func() error { counter++; return nil }}
			c, err := NewConverger(options.Iterative, float64(tt.i), nil)
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.Converge(ctx, s)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if tt.i != counter {
				t.Fatalf("want convergance after %d, but got convergance after %d", tt.i, counter)
			}
		})
	}
}

func TestConverge_Automatic(t *testing.T) {
	tests := []struct {
		name string
		i    int
		t    float64
	}{
		{name: "t=0.1", t: 1e-1, i: 1},
		{name: "t=0.01", t: 1e-2, i: 2},
		// Converge after 4 not 3 iterations because of rounding error.
		{name: "t=0.001", t: 1e-3, i: 4},
		{name: "t=0.0001", t: 1e-4, i: 4},
		// Converge after 6 not 5 iterations because of rounding error.
		{name: "t=0.00001", t: 1e-5, i: 6},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var counter int
			s := stepperMock{takeStep: func() error { counter++; return nil }}
			cost := 10e3
			c, err := NewConverger(options.Automatic, tt.t, func(x [][]float64, y []float64, coeffs []float64) (float64, error) {
				cost -= cost * math.Pow(10, -float64(counter))
				return cost, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.Converge(ctx, s)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if counter != tt.i {
				t.Fatalf("want convergance after %d, but got convergance after %d", tt.i, counter)
			}
		})
	}
}

func TestConverge_StepperErr(t *testing.T) {
	tests := []struct {
		name string
		ct   options.ConverganceType
		err  error
	}{
		{name: "iterative convergance", ct: options.Iterative, err: errors.New("error while trying to converge iteratively")},
		{name: "automatic convergance", ct: options.Automatic, err: errors.New("error while trying to converge automatically")},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := stepperMock{takeStep: func() error { return tt.err }, baseStepper: baseStepper{coeffs: make([]float64, 5)}}
			c, err := NewConverger(tt.ct, 100, func(x [][]float64, y []float64, coeffs []float64) (float64, error) { return 0, nil })
			if err != nil {
				t.Fatal(err)
			}
			_, err = c.Converge(ctx, s)
			if err != tt.err {
				t.Fatalf("want %v, got %v", tt.err, err)
			}
		})
	}
}
