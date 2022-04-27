package regression

import (
	"errors"
	"fmt"
)

var (
	// ErrInvalidFeatureVector indicates that the feature vector is not consistent with trained model.
	ErrInvalidFeatureVector = errors.New("invalid feature vector")
	// ErrNotTrainedModel indicates that the model is not trained.
	ErrNotTrainedModel = errors.New("not trained model")
)

// A Model is a linear regression model.
type Model interface {
	// Predict returns the predicated target value for the given input.
	Predict(x Vector) (float64, error)
	// Coefficients returns the trained linear regression model's coefficients.
	Coefficients() (Vector, error)
	// R2 returns 'R squared'.
	R2() (float64, error)
}

type model struct {
	coeffs Vector
	r2     float64
}

func (m model) Predict(x Vector) (float64, error) {
	if !m.isTrained() {
		return 0, ErrNotTrainedModel
	}
	return calcHypho(x, m.coeffs)
}

func (m model) Coefficients() (Vector, error) {
	if !m.isTrained() {
		return nil, ErrNotTrainedModel
	}
	return m.coeffs, nil
}

func (m model) R2() (float64, error) {
	if !m.isTrained() {
		return 0, ErrNotTrainedModel
	}
	return m.r2, nil
}

func (m model) String() string {
	if !m.isTrained() {
		return ErrNotTrainedModel.Error()
	}
	s := fmt.Sprintf("y = %f", m.coeffs[0])
	for i, coeff := range m.coeffs[1:] {
		s += fmt.Sprintf(" + x%d*%f", i+1, coeff)
	}
	return s
}

func (m model) isTrained() bool {
	return m.coeffs != nil
}

// calcHypho calculates the hyphothesis function.
//
// The hyphothesis equals h(x)=OX, where O stands for a coefficients vector and X is a feature vector.
func calcHypho(x Vector, coeffs Vector) (float64, error) {
	n := len(x)
	if n != len(coeffs)-1 {
		return 0, ErrInvalidFeatureVector
	}
	var y float64
	for i, coeff := range coeffs[1:] {
		y += x[i] * coeff
	}
	return coeffs[0] + y, nil
}
