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

// Result is a gradient descent result.
// It holds calculated coefficients and scaling parameters.
type Result struct {
	Coefficients  []float64
	ScalingResult ScalingResult
}

// New creates new gradient descent.
func New(h Hyphothesis, c CostFunc) GradientDescent {
	return GradientDescent{h, c}
}

// Run runs gradient descent algorithm.
func (g GradientDescent) Run(ctx context.Context, o options.Options, x [][]float64, y []float64) (Result, error) {
	scaler, err := NewScaler(o.FeatureScalingTechnique())
	if err != nil {
		return Result{}, err
	}
	sr := scaler.Scale(x)
	x = sr.X
	gds, err := NewStepper(o.GradientDescentVariant(), g.h, x, y, o.LearningRate())
	if err != nil {
		return Result{}, err
	}
	cv, err := NewConverger(o.ConverganceType(), o.ConverganceIndicator(), g.c)
	if err != nil {
		return Result{}, err
	}
	coeffs, err := cv.Converge(ctx, gds)
	if err != nil {
		return Result{}, err
	}
	return Result{Coefficients: coeffs, ScalingResult: sr}, nil
}
