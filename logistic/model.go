package logistic

import (
	"fmt"
	"math"
)

// A model is a logistic regression model.
type model struct {
	coeffs []float64
	acc    float64
}

func (m model) Predict(x []float64) (int, error) {
	// Include dummy feature equals 1 at the beginning.
	hr, err := hyphothesis(append([]float64{1}, x...), m.coeffs)
	if err != nil {
		return 0, err
	}
	return int(math.Round(hr)), nil
}

func (m model) Coefficients() []float64 {
	coeffs := make([]float64, len(m.coeffs))
	copy(coeffs, m.coeffs)
	return coeffs
}

func (m model) Accuracy() float64 {
	return m.acc
}

func (m model) String() string {
	s := fmt.Sprintf("y = round(%f", m.coeffs[0])
	for i, coeff := range m.coeffs[1:] {
		s += fmt.Sprintf(" + x%d*%f", i+1, coeff)
	}
	return s + ")"
}

func calcAccuracy(x [][]float64, y []float64, coeffs []float64) (float64, error) {
	var correct int
	for i := 0; i < len(x); i++ {
		hr, err := hyphothesis(x[i], coeffs)
		if err != nil {
			return 0, err
		}
		if int(math.Round(hr)) == int(y[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(x)), nil
}
