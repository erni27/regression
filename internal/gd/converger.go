package gd

import (
	"context"

	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

type Converger interface {
	Converge(context.Context, Stepper) ([]float64, error)
}

type ConvergerFunc func(context.Context, Stepper) ([]float64, error)

func (f ConvergerFunc) Converge(ctx context.Context, s Stepper) ([]float64, error) {
	return f(ctx, s)
}

func NewConverger(ct options.ConverganceType, ci float64, c CostFunc) (Converger, error) {
	var cf ConvergerFunc
	switch ct {
	case options.Iterative:
		cf = func(ctx context.Context, s Stepper) ([]float64, error) {
			return convergeAfter(ctx, s, int(ci))
		}
	case options.Automatic:
		cf = func(ctx context.Context, s Stepper) ([]float64, error) {
			return convergeAutomatically(ctx, s, c, ci)
		}
	default:
		return nil, regression.ErrUnsupportedConverganceType
	}
	return cf, nil
}

// convergeAfter runs gradient descent with iterative convergance.
// It converges after i iterations.
func convergeAfter(ctx context.Context, s Stepper, i int) ([]float64, error) {
	for k := 0; k < i; k++ {
		err := s.TakeStep()
		if err != nil {
			return nil, err
		}
	}
	return s.CurrentCoefficients(), nil
}

// convergeAutomatically runs gradient descent with automatic convergance.
// It converges if cost function decreases by lower value than t threshold.
func convergeAutomatically(ctx context.Context, s Stepper, c CostFunc, t float64) ([]float64, error) {
	var coeffs []float64
	for {
		coeffs = s.CurrentCoefficients()
		err := s.TakeStep()
		if err != nil {
			return nil, err
		}
		// Check if cost function decreases by lower value than established threshold.
		oc, err := c(s.X(), s.Y(), coeffs)
		if err != nil {
			return nil, err
		}
		nc, err := c(s.X(), s.Y(), s.CurrentCoefficients())
		if err != nil {
			return nil, err
		}
		if r := 1 - nc/oc; r > 0 && r < t {
			return s.CurrentCoefficients(), nil
		}
	}
}
