package gd

import (
	"math"

	"github.com/erni27/regression"
)

func NewBatchStepper(h Hyphothesis, x [][]float64, y []float64, lr float64) Stepper {
	return &batchStepper{stepper{hypho: h, x: x, y: y, lr: lr, coeffs: make([]float64, len(x[0]))}}
}

// batchStepper takes steps (calculates next values of the coefficients)
// according to the batch gradient descent variant.
// It implements Stepper interface.
type batchStepper struct {
	stepper
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
		pd /= float64(len(s.x))
		// Assign new value to the new coefficients vector.
		nc[j] = s.coeffs[j] + s.lr*pd
		if math.IsNaN(nc[j]) || math.IsInf(nc[j], 0) {
			return regression.ErrCannotConverge
		}
	}
	s.coeffs = nc
	return nil
}
