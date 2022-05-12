package linear

import (
	"fmt"
	"math"

	"github.com/erni27/regression"
)

// A model is a linear regression model.
type model struct {
	coeffs []float64
	r2     float64
}

// Predict returns the predicated target value for the given input.
func (m model) Predict(x []float64) (float64, error) {
	// Include dummy feature equals 1 at the beginning.
	return hyphothesis(append([]float64{1}, x...), m.coeffs)
}

// Coefficients returns the trained linear regression model's coefficients.
func (m model) Coefficients() []float64 {
	return m.coeffs
}

// Accuracy returns 'R squared' determinant for trained model.
func (m model) Accuracy() float64 {
	if m.r2 >= 0 {
		return m.r2
	}
	return m.r2
}

func (m model) String() string {
	s := fmt.Sprintf("y = %f", m.coeffs[0])
	for i, coeff := range m.coeffs[1:] {
		s += fmt.Sprintf(" + x%d*%f", i+1, coeff)
	}
	return s
}

// calcR2 calculates the coefficient of determination (R squared).
func (m model) calcR2(s regression.TrainingSet) (float64, error) {
	var ssr, sst float64
	mr := calcMean(s.GetTargetVector())
	for _, te := range s.Examples() {
		v, err := hyphothesis(te.Features, m.coeffs)
		if err != nil {
			return 0, err
		}
		ssr += math.Pow(te.Target-v, 2)
		sst += math.Pow(te.Target-mr, 2)
	}
	return 1 - ssr/sst, nil
}

func calcMean(y []float64) float64 {
	var s float64
	for _, v := range y {
		s += v
	}
	return s / float64(len(y))
}
