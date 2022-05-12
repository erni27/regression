package logistic

import "math"

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

func calcAccuracy() {

}
