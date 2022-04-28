package regression

import (
	"github.com/erni27/regression/internal/matrix"
)

func Train(ts TrainingSet) (Model, error) {
	m := Model{}
	err := ts.addDummyFeatures()
	if err != nil {
		return m, err
	}

	x := ts.getDesignMatrix()
	y := ts.getTargetVector()

	xt, err := matrix.Transpose(x)
	if err != nil {
		return m, err
	}
	p, err := matrix.Multiply(xt, x)
	if err != nil {
		return m, err
	}
	p, err = matrix.Inverse(p)
	if err != nil {
		return m, err
	}
	p, err = matrix.Multiply(p, xt)
	if err != nil {
		return m, err
	}
	coeffs, err := matrix.MultiplyByVector(p, y)
	if err != nil {
		return m, err
	}

	m.coeffs = coeffs
	return m, nil
}
