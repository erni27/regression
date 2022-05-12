package logistic

import (
	"math"

	"github.com/erni27/regression"
)

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
	return 1 / (1 + math.Pow(math.E, -z)), nil
}
