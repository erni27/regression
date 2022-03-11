package linear

import (
	"errors"

	"github.com/erni27/regression"
)

var (
	// ErrCannotConverge indicates an issue with convergance if learning rate is too big.
	ErrCannotConverge = errors.New("cannot converge")
	// ErrNotTrained indicates that the model is not trained.
	ErrNotTrained = errors.New("model not trained")
	// ErrInvalidFeatureVector indicates that the feature vector is not consistent.
	ErrInvalidFeatureVector = errors.New("invalid feature vector")
)

// Model represents the linear regression model.
type Model struct {
	coefficients regression.Vector
	learningRate float64
}

func NewModel(lr float64) Model {
	return Model{learningRate: lr}
}

func (m *Model) GetLearningRate() float64 {
	return m.learningRate
}

func (m *Model) SetLearningRate(lr float64) {
	m.learningRate = lr
}

// Predict returns the target value of given input.
func (m *Model) Predict(x regression.Vector) (float64, error) {
	if m.coefficients == nil {
		return 0, ErrNotTrained
	}
	return calcHypho(x, m.coefficients)
}

// GetCoefficients gets the trained linear regression model's coefficients.
func (m *Model) GetCoefficients() (regression.Vector, error) {
	if m.coefficients == nil {
		return regression.Vector{}, ErrNotTrained
	}
	return m.coefficients, nil
}

func calcHypho(x regression.Vector, coeff regression.Vector) (float64, error) {
	if len(x)+1 != len(coeff) {
		return 0, ErrInvalidFeatureVector
	}
	var y float64
	y += coeff[0]
	n := len(x)
	for i := 0; i < n; i++ {
		y += x[i] * coeff[i+1]
	}
	return y, nil
}
