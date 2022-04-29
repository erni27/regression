package regression

import (
	"github.com/erni27/regression/internal/matrix"
	"golang.org/x/exp/constraints"
)

// Train runs linear regression for given training set. It uses an analytical approach
// for computing coefficients (normal equation).
func Train[float constraints.Float](ts TrainingSet[float]) (Model[float], error) {
	err := ts.addDummyFeatures()
	if err != nil {
		return Model[float]{}, err
	}
	x := ts.getDesignMatrix()
	y := ts.getTargetVector()

	coeffs, err := solveNormalEquation(x, y)
	if err != nil {
		return Model[float]{}, err
	}
	r2, err := calcR2(ts, coeffs)
	if err != nil {
		return Model[float]{}, err
	}
	return Model[float]{coeffs: coeffs, r2: r2}, nil
}

// solveNormalEquation solves the normal equation for given design matrix and target vector.
//
// The normal equation minimizes the cost function (LMS) by explicity taking its derivatives
// with respect to the coefficients and setting them to zero.
func solveNormalEquation[float constraints.Float](x [][]float, y []float) ([]float, error) {
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
