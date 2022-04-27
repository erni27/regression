package linear

import (
	"errors"

	"github.com/erni27/regression"
)

var (
	// ErrInvalidFeatureVector indicates that the feature vector is not consistent.
	ErrInvalidFeatureVector = errors.New("invalid feature vector")
	// ErrNotTrainedModel indicates that the model is not trained.
	ErrNotTrainedModel = errors.New("not trained model")
)

// TrainingExample represents a single (features, target) example.
type TrainingExample struct {
	Features regression.Vector
	Target   float64
}

// TrainingSet represents set of traning examples.
type TrainingSet []TrainingExample

// A Model is a linear regression model.
type Model interface {
	// Predict returns the predicated target value for the given input.
	Predict(x regression.Vector) (float64, error)
	// GetCoefficients returns the trained linear regression model's coefficients.
	GetCoefficients() (regression.Vector, error)
	// R2 returns 'R squared'.
	R2() (float64, error)
}

// model represents base training model.
type model struct {
	coefficients regression.Vector
	r2           float64
}

func (m model) Predict(x regression.Vector) (float64, error) {
	if m.coefficients == nil {
		return 0, ErrNotTrainedModel
	}
	return calcHypho(x, m.coefficients)
}

func (m model) GetCoefficients() (regression.Vector, error) {
	if m.coefficients == nil {
		return nil, ErrNotTrainedModel
	}
	return m.coefficients, nil
}

func (m model) R2() (float64, error) {
	if m.coefficients == nil {
		return 0, ErrNotTrainedModel
	}
	return m.r2, nil
}

func (m model) String() string {
	return ""
}

// calcHypho calculates the hyphothesis function.
//
// The hyphothesis equals h(x)=OX, where O stands for a coefficients vector and X is a feature vector.
// The dummy feature is added on-fly during the calculation so the input vector should not contain it.
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
