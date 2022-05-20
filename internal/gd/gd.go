// Package gd provides gradient descent implementation.
package gd

import (
	"context"

	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

type Result struct {
	Coefficients []float64
	U            []float64
	S            []float64
}

// Hyphothesis is a function template for a hyphothesis function used in gradient descent algorithm.
type Hyphothesis func(x []float64, coeffs []float64) (float64, error)

// CostFunc is a function template for cost function used in gradient descent algorithm.
type CostFunc func(x [][]float64, y []float64, coeffs []float64) (float64, error)

// GradientDescent holds the hyphothesis and cost functions needed by gradient descent algorithm.
type GradientDescent struct {
	h Hyphothesis
	c CostFunc
}

// New creates new gradient descent.
func New(h Hyphothesis, c CostFunc) GradientDescent {
	return GradientDescent{h, c}
}

// Run runs gradient descent algorithm.
func (g GradientDescent) Run(ctx context.Context, o options.Options, x [][]float64, y []float64) (Result, error) {
	// Scale features.
	scaler, err := NewScaler(o.FeatureScalingTechnique())
	if err != nil {
		return Result{}, err
	}
	sr := scaler.Scale(x)
	x = sr.X
	res := Result{U: sr.U, S: sr.S}

	// Init stepper.
	var gds stepper
	switch o.GradientDescentVariant() {
	case options.Batch:
		gds = newBatchStepper(g.h, x, y, o.LearningRate())
	case options.Stochastic:
		gds = newStochasticStepper(g.h, x, y, o.LearningRate())
	default:
		return Result{}, regression.ErrUnsupportedGradientDescentVariant
	}

	var coeffs []float64
	switch o.ConverganceType() {
	case options.Iterative:
		coeffs, err = convergeAfter(gds, int(o.ConverganceIndicator()))
		if err != nil {
			return Result{}, err
		}
	case options.Automatic:
		coeffs, err = convergeAutomatically(gds, g.c, o.ConverganceIndicator())
		if err != nil {
			return Result{}, err
		}
	default:
		return Result{}, regression.ErrUnsupportedConverganceType
	}
	res.Coefficients = coeffs
	return res, nil
}

// A stepper wraps logic around taking steps (calculating new coefficients' values).
type stepper interface {
	// TakeStep takes single step towards cost function minimum.
	TakeStep() error
	// CurrentCoefficients returns current coefficients calculated by stepper.
	CurrentCoefficients() []float64
	// X returns design matrix used in calculations.
	X() [][]float64
	// Y returns target vector used in calculations.
	Y() []float64
}

// convergeAfter runs gradient descent with iterative convergance.
// It converges after i iterations.
func convergeAfter(s stepper, i int) ([]float64, error) {
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
func convergeAutomatically(s stepper, c CostFunc, t float64) ([]float64, error) {
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
