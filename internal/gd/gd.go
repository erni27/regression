// Package gd provides gradient descent implementation.
package gd

import "github.com/erni27/regression"

// A Stepper wraps logic around taking steps (calculating new coefficients' values).
type Stepper interface {
	// TakeStep takes single step towards cost function minimum.
	TakeStep() error
	// CurrentCoefficients returns current coefficients calculated by stepper.
	CurrentCoefficients() []float64
	// X returns design matrix used in calculations.
	X() [][]float64
	// Y returns target vector used in calculations.
	Y() []float64
}

// Hyphothesis is a function template for a hyphothesis function used in gradient descent algorithm.
type Hyphothesis func(x []float64, coeffs []float64) (float64, error)

// CostFunc is a function template for cost function used in gradient descent algorithm.
type CostFunc func(x [][]float64, y []float64, coeffs []float64) (float64, error)

func ConvergeAfter(s Stepper, i int) ([]float64, error) {
	for k := 0; k < i; k++ {
		err := s.TakeStep()
		if err != nil {
			return nil, err
		}
	}
	return s.CurrentCoefficients(), nil
}

func ConvergeAutomatically(s Stepper, c CostFunc, t float64) ([]float64, error) {
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
		if nc/oc > 1 {
			return nil, regression.ErrCannotConverge
		}
		if 1-nc/oc < t {
			return s.CurrentCoefficients(), nil
		}
	}
}

// stepper is a prototype for concrete steppers. It should be embedded.
type stepper struct {
	hypho  Hyphothesis
	x      [][]float64
	y      []float64
	lr     float64
	coeffs []float64
}

func (s stepper) CurrentCoefficients() []float64 {
	return s.coeffs
}

func (s stepper) X() [][]float64 {
	return s.x
}

func (s stepper) Y() []float64 {
	return s.y
}
