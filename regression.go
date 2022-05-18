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
	// ErrInvalidTrainingSet is returned if features vectors included in the set are not consistent.
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

// TrainingExample represents a single (features, target) example.
type TrainingExample struct {
	Features []float64
	Target   float64
}

// NewTrainingSet creates a new training set if examples are consistent.
// If examples are not consistent, returns an error. It doesn't check for correctness of values.
func NewTrainingSet(exs []TrainingExample) (TrainingSet, error) {
	if !areExamplesConsistent(exs) {
		return TrainingSet{}, ErrInvalidTrainingSet
	}
	return TrainingSet{examples: exs}, nil
}

// TrainingSet represents set of traning examples.
type TrainingSet struct {
	examples     []TrainingExample
	x            [][]float64
	y            []float64
	isDummyAdded bool // Indicates if a dummy feature was added to each training example.
}

// Examples returns all training examples from training set.
func (s *TrainingSet) Examples() []TrainingExample {
	return s.examples
}

// GetTargetVector returns the target vector from the training set.
func (s *TrainingSet) GetTargetVector() []float64 {
	if s.y != nil {
		return s.y
	}
	y := make([]float64, len(s.examples))
	for i, te := range s.examples {
		y[i] = te.Target
	}
	s.y = y
	return y
}

// GetDesignMatrix returns the design matrix from the training set.
func (s *TrainingSet) GetDesignMatrix() [][]float64 {
	if s.x != nil {
		return s.x
	}
	x := make([][]float64, len(s.examples))
	for i, te := range s.examples {
		x[i] = te.Features
	}
	s.x = x
	return x
}

// AddDummyFeatures puts dummy feature (equals 1) for each training example being a part of the training set.
func (s *TrainingSet) AddDummyFeatures() {
	if s.isDummyAdded {
		return
	}
	for i := 0; i < len(s.examples); i++ {
		s.examples[i].Features = append([]float64{1}, s.examples[i].Features...)
	}
	s.isDummyAdded = true
}

func areExamplesConsistent(exs []TrainingExample) bool {
	if len(exs) == 0 {
		return false
	}
	// n represents number of features.
	// This number must be consistent across all features vectors.
	n := len(exs[0].Features)
	for i := 1; i < len(exs); i++ {
		if len(exs[i].Features) != n {
			return false
		}
	}
	return true
}
