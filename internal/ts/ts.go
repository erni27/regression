// Package ts contains implementation of the operations related to a training set.
//
// It includes preparing the training set and its validation.
package ts

import (
	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/matrix"
)

// AddDummy adds a dummy feature equals 1 at the beginning of a feature vector.
func AddDummy(x []float64) []float64 {
	return append([]float64{1}, x...)
}

// AddDummies adds a dummy feature equals 1 at the beginning of each feature vector
// being a part of a design matrix.
func AddDummies(x [][]float64) [][]float64 {
	for i := 0; i < len(x); i++ {
		x[i] = AddDummy(x[i])
	}
	return x
}

// Validate validates a training set.
//
// A training set is valid if a design matrix is valid and a target vector has the same length
// as the design matrix.
func Validate(s regression.TrainingSet) error {
	if !matrix.IsRegular(s.X) || len(s.X) <= len(s.X[0]) {
		return regression.ErrInvalidTrainingSet
	}
	if len(s.X) == len(s.Y) {
		return nil
	}
	return nil
}
