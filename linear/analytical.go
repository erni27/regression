package linear

import (
	"context"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/long"
	"github.com/erni27/regression/internal/matrix"
)

// WithNormalEquation initializes linear regression with analytical approach.
// It directly finds the value of coefficients by solving normal equation.
func WithNormalEquation() regression.Regression[float64] {
	var f regression.RegressionFunc[float64] = analytical
	return f
}

// analytical runs linear regression for given training set. It uses an analytical approach
// for computing coefficients (normal equation).
func analytical(ctx context.Context, s regression.TrainingSet) (regression.Model[float64], error) {
	if !matrix.IsRegular(s.X) || len(s.X) != len(s.Y) {
		return nil, regression.ErrInvalidTrainingSet
	}
	x := matrix.AddDummies(s.X)
	y := s.Y
	coeffs, err := solveNormalEquation(ctx, x, y)
	if err != nil {
		return nil, err
	}
	r2, err := calcR2(x, y, coeffs)
	if err != nil {
		return nil, err
	}
	return model{coeffs: coeffs, r2: r2}, nil
}

// solveNormalEquation solves the normal equation for given design matrix and target vector.
//
// The normal equation minimizes the cost function for linear regression (LMS) by explicity taking its derivatives
// with respect to the coefficients and setting them to zero.
func solveNormalEquation(ctx context.Context, x [][]float64, y []float64) ([]float64, error) {
	xt, err := matrix.Transpose(ctx, x)
	if err != nil {
		return nil, err
	}
	p, err := long.Run(ctx, func() ([][]float64, error) { return matrix.Multiply(ctx, xt, x) })
	if err != nil {
		return nil, err
	}
	p, err = long.Run(ctx, func() ([][]float64, error) { return matrix.Inverse(ctx, p) })
	if err != nil {
		return nil, err
	}
	p, err = long.Run(ctx, func() ([][]float64, error) { return matrix.Multiply(ctx, p, xt) })
	if err != nil {
		return nil, err
	}
	o, err := long.Run(ctx, func() ([]float64, error) { return matrix.MultiplyByVector(ctx, p, y) })
	if err != nil {
		return nil, err
	}
	return o, nil
}
