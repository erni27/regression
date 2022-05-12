package linear

import (
	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/gd"
	"github.com/erni27/regression/options"
)

// WithGradientDescent initializes linear regression with numerical approach.
// It finds the value of coefficients by taking steps in each iteration towards
// the minimum of a cost function (LMS).
func WithGradientDescent(o options.Options) regression.Regression[float64] {
	var f regressionFunc = func(s regression.TrainingSet) (regression.Model[float64], error) {
		return numerical(o, s)
	}
	return f
}

// numerical runs linear regression for given training set. It uses an numerical approach
// for computing coefficients (gradient descent).
func numerical(o options.Options, s regression.TrainingSet) (regression.Model[float64], error) {
	s.AddDummyFeatures()
	x := s.GetDesignMatrix()
	y := s.GetTargetVector()

	coeffs, err := gd.Run(o, hyphothesis, cost, x, y)
	if err != nil {
		return nil, err
	}

	r2, err := calcR2(s, coeffs)
	if err != nil {
		return nil, err
	}
	return model{coeffs: coeffs, r2: r2}, nil
}
