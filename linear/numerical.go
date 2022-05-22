package linear

import (
	"context"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/gd"
	"github.com/erni27/regression/internal/long"
	"github.com/erni27/regression/internal/matrix"
	"github.com/erni27/regression/options"
)

var gradientDescent gd.GradientDescent = gd.New(hyphothesis, cost)

// WithGradientDescent initializes linear regression with numerical approach.
// It finds the value of coefficients by taking steps in each iteration towards
// the minimum of a cost function (LMS).
func WithGradientDescent(o options.Options) regression.Regression[float64] {
	var f regression.RegressionFunc[float64] = func(ctx context.Context, s regression.TrainingSet) (regression.Model[float64], error) {
		return numerical(ctx, o, s)
	}
	return f
}

// numerical runs linear regression for given training set. It uses an numerical approach
// for computing coefficients (gradient descent).
func numerical(ctx context.Context, o options.Options, s regression.TrainingSet) (regression.Model[float64], error) {
	if !matrix.IsRegular(s.X) || len(s.X) != len(s.Y) {
		return nil, regression.ErrInvalidTrainingSet
	}
	x := matrix.AddDummies(s.X)
	y := s.Y
	coeffs, err := long.Run(ctx, func() ([]float64, error) { return gradientDescent.Run(ctx, o, x, y) })
	if err != nil {
		return nil, err
	}
	r2, err := calcR2(x, y, coeffs)
	if err != nil {
		return nil, err
	}
	return model{coeffs: coeffs, r2: r2}, nil
}
