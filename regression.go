// Package regression defines interfaces, structures and errors shared by other packages
// that implement a concrete regression algorithm.
package regression

import (
	"context"
	"errors"
)

var (
	// ErrCannotConverge is returned if gradient descent cannot converge.
	// It usually means that the learning rate is too large.
	ErrCannotConverge = errors.New("cannot converge")
	// ErrUnsupportedGradientDescentVariant is returned if unsupported gradient descent variant was chosen.
	ErrUnsupportedGradientDescentVariant = errors.New("unsupported gradient descent variant")
	// ErrUnsupportedConverganceType is returned if unsupported convergance type was chosen.
	ErrUnsupportedConverganceType = errors.New("unsupported convergance type")
	// ErrUnsupportedScalingTechnique is returned if unsupported features' scaling technique was chosen.
	ErrUnsupportedScalingTechnique = errors.New("unsupported scaling technique")
	// ErrInvalidTrainingSet is returned if a design matrix is invalid or doesn't have the same length as a target vector.
	ErrInvalidTrainingSet = errors.New("invalid training set")
	// ErrInvalidFeatureVector is returned if feature vector is invalid.
	ErrInvalidFeatureVector = errors.New("invalid feature vector")
)

// TargetType is a constraint that permits two types (float64 or integer) for target value.
// Floating point numbers are used for continous value of y, while integer corresponds to the
// discrete one.
type TargetType interface {
	~float64 | ~int
}

// A Model is a trained regression model.
type Model[T TargetType] interface {
	// Predict returns the predicated target value for the given input.
	Predict([]float64) (T, error)
	// Coefficients returns the trained regression model's coefficients.
	Coefficients() []float64
	// Accuracy returns calculated accuracy for trained model.
	Accuracy() float64
}

// A Regression is a regression runner. It provides an abstraction for model training.
type Regression[T TargetType] interface {
	// Run runs regression against input training set.
	// It returns trained Model if succeded, otherwise returns an error.
	Run(context.Context, TrainingSet) (Model[T], error)
}

// RegressionFunc is an adapter to allow the use of plain functions as regressions.
type RegressionFunc[T TargetType] func(context.Context, TrainingSet) (Model[T], error)

// Run calls f(s).
func (f RegressionFunc[T]) Run(ctx context.Context, s TrainingSet) (Model[T], error) {
	return f(ctx, s)
}

// TrainingSet represents a set of traning examples.
type TrainingSet struct {
	// X is a design matrix.
	X [][]float64
	// Y is a target vector.
	Y []float64
}
