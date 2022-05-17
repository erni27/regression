package linear

import (
	"fmt"
	"math"
)

// A model is a linear regression model.
type model struct {
	coeffs []float64
	r2     float64
}

func (m model) Predict(x []float64) (float64, error) {
	// Include dummy feature equals 1 at the beginning.
	return hyphothesis(append([]float64{1}, x...), m.coeffs)
}

func (m model) Coefficients() []float64 {
	return m.coeffs
}

func (m model) Accuracy() float64 {
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
func calcR2(x [][]float64, y, coeffs []float64) (float64, error) {
	var ssr, sst float64
	mr := calcMean(y)
	for i := 0; i < len(x); i++ {
		v, err := hyphothesis(x[i], coeffs)
		if err != nil {
			return 0, err
		}
		ssr += math.Pow(y[i]-v, 2)
		sst += math.Pow(y[i]-mr, 2)
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
