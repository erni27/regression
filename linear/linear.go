// Package linear provides linear regression model implementation.
package linear

import (
	"math"

	"github.com/erni27/regression"
)

// regressionFunc is an adapter to allow the use of plain functions as regressions.
type regressionFunc func(s regression.TrainingSet) (regression.Model[float64], error)

// Run calls f(s).
func (f regressionFunc) Run(s regression.TrainingSet) (regression.Model[float64], error) {
	return f(s)
}

// hyphothesis calculates the hyphothesis function for the linear regression model.
//
// The hyphothesis equals h(x)=OX, where O stands for a coefficients vector and X is a feature vector.
func hyphothesis(x []float64, coeffs []float64) (float64, error) {
	if len(x) != len(coeffs) {
		return 0, regression.ErrInvalidFeatureVector
	}
	var y float64
	for i := 0; i < len(coeffs); i++ {
		y += x[i] * coeffs[i]
	}
	return y, nil
}

// cost calculates cost function value for linear regression.
func cost(x [][]float64, y []float64, coeffs []float64) (float64, error) {
	m := len(x)
	var c float64
	for i := 0; i < m; i++ {
		hr, err := hyphothesis(x[i], coeffs)
		if err != nil {
			return 0, err
		}
		c += math.Pow(hr-y[i], 2)
	}
	return c / float64((2 * m)), nil
}
