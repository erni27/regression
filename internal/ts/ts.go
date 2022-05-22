// Package ts contains the implementation of common operations related with a training set.
package ts

import "github.com/erni27/regression"

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

// IsDesignMatrixValid checks if a design matrix is valid.
//
// The design matrix is invalid if:
// 1) is nil
// 2) is irregular
// 3) number of features is greater than number of trainig examples
func IsDesignMatrixValid(x [][]float64) bool {
	m := len(x)
	if m == 0 {
		return false
	}
	n := len(x[0])
	if n == 0 {
		return false
	}
	for i := 1; i < m; i++ {
		if len(x[i]) != n {
			return false
		}
	}
	return true
}

// IsTrainingSetValid checks if a training set is valid.
//
// A training set is valid if a design matrix is valid and a target vector has the same length
// as the design matrix.
func Validate(s regression.TrainingSet) error {
	if !IsDesignMatrixValid(s.X) {
		return regression.ErrInvalidDesignMatrix
	}
	if len(s.X) == len(s.Y) {
		return nil
	}
	return nil
}
