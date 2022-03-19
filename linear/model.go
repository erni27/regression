package linear

import (
	"errors"

	"github.com/erni27/regression"
)

var (
	// ErrInvalidFeatureVector indicates that the feature vector is not consistent.
	ErrInvalidFeatureVector = errors.New("invalid feature vector")
	// ErrInvalidModelType indicates that the input model is invalid.
	ErrInvalidModelType = errors.New("invalid model type")
	// ErrNotTrainedModel indicates that the model is not trained.
	ErrNotTrainedModel = errors.New("not trained model")
)

type optimisationType int

// Discriminators that allow distinguish between optimisation type.
const (
	batchGd optimisationType = iota + 1
	stochasticGd
	normalEquation
)

// A Model is a linear regression model.
type Model interface {
	// Predict returns the predicated target value for the given input.
	Predict(x regression.Vector) (float64, error)
	// GetCoefficients returns the trained linear regression model's coefficients.
	GetCoefficients() (regression.Vector, error)
	// R2 returns 'R squared'.
	R2() (float64, error)
}

// NewModel returns a non-nil, base linear regression model.
// By default, base model passed to the Train function uses
// an analytical approach (normal equation) to find the coefficients.
func NewModel(learningRate float64) Model {
	return model{learningRate: learningRate, optimisationType: normalEquation}
}

// WithBatchGradientDescent returns a batch gradient descent model.
// Passed to the Train function uses an iterative approach (batch gradient descent)
// to find the coefficients. n indicates how many times the loop is invoked.
func WithBatchGradientDescent(m Model, n int) (Model, error) {
	b, err := getBaseModel(m)
	if err != nil {
		return model{}, err
	}
	b.optimisationType = batchGd
	return gdModel{model: b, iterations: n}, nil
}

// WithStochasticGradientDescent returns a stochastic gradient descent model.
// Passed to the Train function uses an iterative approach (stochastic gradient descent)
// to find the coefficients. n indicates how many times the loop is invoked.
func WithStochasticGradientDescent(m Model, n int) (Model, error) {
	b, err := getBaseModel(m)
	if err != nil {
		return model{}, err
	}
	b.optimisationType = stochasticGd
	return gdModel{model: b, iterations: n}, nil
}

// WithAutomaticConvergence enriches the gradient descent model with an
// automatic convergence test.
// Automatic convergence test declares convergance if cost function decreases
// by less than t.
func WithAutomaticConvergence(m Model, t float64) (Model, error) {
	b, err := getBaseModel(m)
	if err != nil {
		return model{}, err
	}
	if b.optimisationType == normalEquation {
		return model{}, ErrInvalidModelType
	}
	return gdAutoConvModel{model: b, threshold: t}, nil
}

func getBaseModel(parent Model) (model, error) {
	switch m := parent.(type) {
	case model:
		return m, nil
	case gdModel:
		return m.model, nil
	case gdAutoConvModel:
		return m.model, nil
	default:
		return model{}, ErrInvalidModelType
	}
}

// model represents base training model.
type model struct {
	learningRate     float64
	coefficients     regression.Vector
	r2               float64
	optimisationType optimisationType
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

// calcHypho calculates the hyphothesis function.
//
// The hyphothesis equals h(x)=OX, where O stands for a coefficients vector
// and X is a feature vector (with dummy feature at the first position equals 1).
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

// gdModel is a gradient descent model that carries the information
// about the number of iterations before the algorithm converge.
type gdModel struct {
	model
	iterations int
}

// gdModel is a gradient descent model that carries
// the information about the automatic convergence test threshold.
type gdAutoConvModel struct {
	model
	threshold float64
}
