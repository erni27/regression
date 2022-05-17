package gd

import (
	"math"

	"github.com/erni27/regression"
)

// newStochasticStepper returns a new Stepper which uses stochastic gradient descent algorithm
// to calculate next steps.
func newStochasticStepper(h Hyphothesis, x [][]float64, y []float64, lr float64) stepper {
	return &stochasticStepper{baseStepper{hypho: h, x: x, y: y, lr: lr, coeffs: make([]float64, len(x[0]))}, 0}
}

// stochasticStepper takes steps (calculates next values of the coefficients)
// according to the stochastic gradient descent variant.
// It implements Stepper interface.
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
