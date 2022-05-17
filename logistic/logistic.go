package logistic

import (
	"math"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/gd"
	"github.com/erni27/regression/options"
)

// WithGradientDescent initializes logistic regression with numerical approach.
// It finds the value of coefficients by taking steps in each iteration towards
// the minimum of a cost function.
func WithGradientDescent(o options.Options) regression.Regression[int] {
	var f regression.RegressionFunc[int] = func(s regression.TrainingSet) (regression.Model[int], error) {
		return numerical(o, s)
	}
	return f
}

// run runs logistic regression for given training set. It uses an numerical approach
// for computing coefficients (gradient descent).
func numerical(o options.Options, s regression.TrainingSet) (regression.Model[int], error) {
	s.AddDummyFeatures()
	x := s.GetDesignMatrix()
	y := s.GetTargetVector()

	coeffs, err := gd.Run(o, hyphothesis, cost, x, y)
	if err != nil {
		return nil, err
	}
	acc, err := calcAccuracy(x, y, coeffs)
	if err != nil {
		return nil, err
	}
	return model{coeffs: coeffs, acc: acc}, nil
}

// hyphothesis calculates the hyphothesis function for the logistic regression model.
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

// cost calculates cost function value for logistic regression.
func cost(x [][]float64, y []float64, coeffs []float64) (float64, error) {
	panic("not implemented")
}
