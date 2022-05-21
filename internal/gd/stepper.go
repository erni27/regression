package gd

import (
	"math"

	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

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

func NewStepper(gdv options.GradientDescentVariant, h Hyphothesis, x [][]float64, y []float64, lr float64) (Stepper, error) {
	var gds Stepper
	switch gdv {
	case options.Batch:
		gds = &batchStepper{baseStepper{hypho: h, x: x, y: y, lr: lr, coeffs: make([]float64, len(x[0]))}}
	case options.Stochastic:
		gds = &stochasticStepper{baseStepper{hypho: h, x: x, y: y, lr: lr, coeffs: make([]float64, len(x[0]))}, 0}
	default:
		return nil, regression.ErrUnsupportedGradientDescentVariant
	}
	return gds, nil
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

// batchStepper takes steps (calculates next values of the coefficients) according to the batch gradient descent variant.
type batchStepper struct {
	baseStepper
}

func (s *batchStepper) TakeStep() error {
	nc := make([]float64, len(s.coeffs))
	for j := 0; j < len(s.coeffs); j++ {
		// Calculate partial derivative.
		var pd float64
		for i := 0; i < len(s.x); i++ {
			hr, err := s.hypho(s.x[i], s.coeffs)
			if err != nil {
				return err
			}
			pd += (s.y[i] - hr) * s.x[i][j]
		}
		// Assign new value to the new coefficients vector.
		nc[j] = s.coeffs[j] + s.lr*pd
		if math.IsNaN(nc[j]) || math.IsInf(nc[j], 0) {
			return regression.ErrCannotConverge
		}
	}
	s.coeffs = nc
	return nil
}

// stochasticStepper takes steps (calculates next values of the coefficients) according to the stochastic gradient descent variant.
type stochasticStepper struct {
	baseStepper
	i int
}

func (s *stochasticStepper) TakeStep() error {
	nc := make([]float64, len(s.coeffs))
	for j := 0; j < len(s.coeffs); j++ {
		hr, err := s.hypho(s.x[s.i], s.coeffs)
		if err != nil {
			return err
		}
		nc[j] = s.coeffs[j] + s.lr*(s.y[s.i]-hr)*s.x[s.i][j]
		if math.IsNaN(nc[j]) || math.IsInf(nc[j], 0) {
			return regression.ErrCannotConverge
		}
	}
	s.i++
	if s.i == len(s.y) {
		s.i = 0
	}
	s.coeffs = nc
	return nil
}
