package regression

import (
	"errors"
	"fmt"

	"golang.org/x/exp/constraints"
)

var (
	ErrInvalidFeatureVector = errors.New("invalid feature vector")
	ErrNotTrainedModel      = errors.New("not trained model")
)

// A Model is a linear regression model.
type Model[float constraints.Float] struct {
	coeffs []float
	r2     float
}

// Predict returns the predicated target value for the given input.
func (m Model[float]) Predict(x []float) (float, error) {
	if !m.IsTrained() {
		return 0, ErrNotTrainedModel
	}
	return calcHypho(x, m.coeffs)
}

// Coefficients returns the trained linear regression model's coefficients.
func (m Model[float]) Coefficients() ([]float, error) {
	if !m.IsTrained() {
		return nil, ErrNotTrainedModel
	}
	return m.coeffs, nil
}

// R2 returns 'R squared'.
func (m Model[float]) R2() (float, error) {
	if !m.IsTrained() {
		return 0, ErrNotTrainedModel
	}
	return m.r2, nil
}

func (m Model[_]) String() string {
	if !m.IsTrained() {
		return ErrNotTrainedModel.Error()
	}
	s := fmt.Sprintf("y = %f", m.coeffs[0])
	for i, coeff := range m.coeffs[1:] {
		s += fmt.Sprintf(" + x%d*%f", i+1, coeff)
	}
	return s
}

// IsTrained checks if linear regression model is trained.
func (m Model[_]) IsTrained() bool {
	return m.coeffs != nil
}

// calcHypho calculates the hyphothesis function.
//
// The hyphothesis equals h(x)=OX, where O stands for a coefficients vector and X is a feature vector.
// It includes dummy feature during the calculation.
func calcHypho[float constraints.Float](x []float, coeffs []float) (float, error) {
	n := len(x)
	if n != len(coeffs)-1 {
		return 0, ErrInvalidFeatureVector
	}
	var y float
	for i, coeff := range coeffs[1:] {
		y += x[i] * coeff
	}
	return coeffs[0] + y, nil
}

func calcMean[float constraints.Float](y []float) float {
	var s float
	for _, v := range y {
		s += v
	}
	return s / float(len(y))
}
