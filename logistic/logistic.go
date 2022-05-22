package logistic

import (
	"context"
	"math"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/gd"
	"github.com/erni27/regression/internal/long"
	"github.com/erni27/regression/internal/matrix"
	"github.com/erni27/regression/options"
)

var gradientDescent gd.GradientDescent = gd.New(hyphothesis, cost)

// WithGradientDescent initializes logistic regression with numerical approach.
// It finds the value of coefficients by taking steps in each iteration towards
// the minimum of a cost function.
func WithGradientDescent(o options.Options) regression.Regression[int] {
	var f regression.RegressionFunc[int] = func(ctx context.Context, s regression.TrainingSet) (regression.Model[int], error) {
		return run(ctx, o, s)
	}
	return f
}

// run runs logistic regression for given training set. It uses an numerical approach
// for computing coefficients (gradient descent).
func run(ctx context.Context, o options.Options, s regression.TrainingSet) (regression.Model[int], error) {
	if !matrix.IsRegular(s.X) || len(s.X) != len(s.Y) || len(s.X) < len(s.X[0]) {
		return nil, regression.ErrInvalidTrainingSet
	}
	x := matrix.AddDummies(s.X)
	y := s.Y
	coeffs, err := long.Run(ctx, func() ([]float64, error) { return gradientDescent.Run(ctx, o, x, y) })
	if err != nil {
		return nil, err
	}
	acc, err := calcAccuracy(x, y, coeffs)
	if err != nil {
		return nil, err
	}
	return model{coeffs: coeffs, acc: acc}, nil
}

// hyphothesis calculates a hyphothesis function value for the logistic regression model.
//
// The hyphothesis equals h(x)=g(OX), where g(z) is a sigmoid function equals g(z)=1/(1-e^(-z))
// and O stands for a coefficients vector and X is a feature vector.
func hyphothesis(x []float64, coeffs []float64) (float64, error) {
	if len(x) != len(coeffs) {
		return 0, regression.ErrInvalidFeatureVector
	}
	var z float64
	for i := 0; i < len(coeffs); i++ {
		z += x[i] * coeffs[i]
	}
	return 1 / (1 + math.Exp(-z)), nil
}

// cost calculates a cost function value for the logistic regression.
func cost(x [][]float64, y []float64, coeffs []float64) (float64, error) {
	m := len(x)
	var c float64
	for i := 0; i < m; i++ {
		hr, err := hyphothesis(x[i], coeffs)
		if err != nil {
			return 0, err
		}
		c += -y[i]*math.Log(hr) - (1-y[i])*math.Log(1-hr)
	}
	return c / float64(m), nil
}
