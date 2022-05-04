package linear

import (
	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/matrix"
)

// WithNormalEquation initializes linear regression with analytical approach.
// It directly finds the value of coefficients by solving normal equation.
func WithNormalEquation() regression.Regression[float64] {
	var f regressionFunc = analytical
	return f
}

// analytical runs linear regression for given training set. It uses an analytical approach
// for computing coefficients (normal equation).
func analytical(s regression.TrainingSet) (regression.Model[float64], error) {
	s.AddDummyFeatures()
	x := s.GetDesignMatrix()
	y := s.GetTargetVector()

	coeffs, err := solveNormalEquation(x, y)
	if err != nil {
		return nil, err
	}
	r2, err := calcR2(s, coeffs)
	if err != nil {
		return nil, err
	}
	return model{coeffs: coeffs, r2: r2}, nil
}

// solveNormalEquation solves the normal equation for given design matrix and target vector.
//
// The normal equation minimizes the cost function for linear regression (LMS) by explicity taking its derivatives
// with respect to the coefficients and setting them to zero.
func solveNormalEquation(x [][]float64, y []float64) ([]float64, error) {
	xt, err := matrix.Transpose(x)
	if err != nil {
		return nil, err
	}
	p, err := matrix.Multiply(xt, x)
	if err != nil {
		return nil, err
	}
	p, err = matrix.Inverse(p)
	if err != nil {
		return nil, err
	}
	p, err = matrix.Multiply(p, xt)
	if err != nil {
		return nil, err
	}
	o, err := matrix.MultiplyByVector(p, y)
	if err != nil {
		return nil, err
	}
	return o, nil
}
