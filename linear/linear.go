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

// hyphothesis calculates the hyphothesis function for linear regression.
//
// The hyphothesis equals h(x)=OX, where O stands for a coefficients vector and X is a feature vector.
func hyphothesis(x []float64, coeffs []float64) (float64, error) {
	if len(x) != len(coeffs) {
		return 0, ErrInvalidFeatureVector
	}
	var y float64
	for i, c := range coeffs {
		y += x[i] * c
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

// addDummyFeatures puts dummy feature (equals 1) for each training example being a part of the training set.
//
// If addDummyFeatures finds inconsistency in the training set, it also returns an error.
func addDummyFeatures(ts *regression.TrainingSet) error {
	// n represents the number of features
	n := len((*ts)[0].Features)
	for i := 0; i < len(*ts); i++ {
		if n != len((*ts)[i].Features) {
			return ErrInvalidTrainingSet
		}
		(*ts)[i].Features = append([]float64{1}, (*ts)[i].Features...)
	}
	return nil
}
