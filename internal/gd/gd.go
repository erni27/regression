// Package gd provides gradient descent implementation.
package gd

import (
	"context"

	"github.com/erni27/regression/options"
)

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

// Run runs the gradient descent algorithm.
func (g GradientDescent) Run(ctx context.Context, o options.Options, x [][]float64, y []float64) ([]float64, error) {
	gds, err := NewStepper(o.GradientDescentVariant(), g.h, x, y, o.LearningRate())
	if err != nil {
		return nil, err
	}
	cv, err := NewConverger(o.ConvergenceType(), o.ConvergenceIndicator(), g.c)
	if err != nil {
		return nil, err
	}
	return cv.Converge(ctx, gds)
}
