// Package gd provides gradient descent implementation.
package gd

import (
	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

// Hyphothesis is a function template for a hyphothesis function used in gradient descent algorithm.
type Hyphothesis func(x []float64, coeffs []float64) (float64, error)

// CostFunc is a function template for cost function used in gradient descent algorithm.
type CostFunc func(x [][]float64, y []float64, coeffs []float64) (float64, error)

// Run runs gradient descent algorithm.
func Run(o options.Options, h Hyphothesis, c CostFunc, x [][]float64, y []float64) ([]float64, error) {
	// Init stepper.
	var gds stepper
	switch o.GradientDescentVariant() {
	case options.Batch:
		gds = newBatchStepper(h, x, y, o.LearningRate())
	case options.Stochastic:
		gds = newStochasticStepper(h, x, y, o.LearningRate())
	default:
		return nil, regression.ErrUnsupportedGradientDescentVariant
	}

	switch o.ConverganceType() {
	case options.Iterative:
		return convergeAfter(gds, int(o.ConverganceIndicator()))
	case options.Automatic:
		return convergeAutomatically(gds, c, o.ConverganceIndicator())
	default:
		return nil, regression.ErrUnsupportedConverganceType
	}
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

// baseStepper is a prototype for concrete steppers. It should be embedded.
type baseStepper struct {
	hypho  Hyphothesis
	x      [][]float64
	y      []float64
	lr     float64
	coeffs []float64
}

func (s baseStepper) CurrentCoefficients() []float64 {
	return s.coeffs
}

func (s baseStepper) X() [][]float64 {
	return s.x
}

func (s baseStepper) Y() []float64 {
	return s.y
}
