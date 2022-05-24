package gd

import (
	"context"

	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

// Converger is the interface that wraps the basic Converge method.
type Converger interface {
	Converge(context.Context, Stepper) ([]float64, error)
}

// ConvergerFunc is an adapter to allow the use of plain functions as convergers.
type ConvergerFunc func(context.Context, Stepper) ([]float64, error)

func (f ConvergerFunc) Converge(ctx context.Context, s Stepper) ([]float64, error) {
	return f(ctx, s)
}

// NewConverger returns a new converger. If unsupported ConvergenceType is passed, an error is returned.
func NewConverger(ct options.ConvergenceType, ci float64, c CostFunc) (Converger, error) {
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
		return nil, regression.ErrUnsupportedConvergenceType
	}
	return cf, nil
}

// convergeAfter runs gradient descent with iterative convergence.
// It converges after i iterations.
func convergeAfter(ctx context.Context, s Stepper, i int) ([]float64, error) {
	for k := 0; k < i; k++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			err := s.TakeStep()
			if err != nil {
				return nil, err
			}
		}
	}
	return s.CurrentCoefficients(), nil
}

// convergeAutomatically runs gradient descent with automatic convergence.
// It converges if cost function decreases by lower value than t threshold.
func convergeAutomatically(ctx context.Context, s Stepper, c CostFunc, t float64) ([]float64, error) {
	var coeffs []float64
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
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
}
