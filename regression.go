package regression

import "errors"

var (
	ErrInvalidTrainingSet = errors.New("invalid training set")
)

// TrainingExample represents a single (features, target) example.
type TrainingExample struct {
	Features []float64
	Target   float64
}

// TrainingSet represents set of traning examples.
type TrainingSet []TrainingExample

// addDummyFeatures puts dummy feature (equals 1) for each training example being a part of the training set.
//
// If addDummyFeatures finds inconsistency in the training set, it also returns an error.
func (ts *TrainingSet) addDummyFeatures() error {
	// n represents the number of features
	n := len((*ts)[0].Features)
	for i := 0; i < len(*ts); i++ {
		if n != len((*ts)[i].Features) {
			return ErrInvalidTrainingSet
		}
		(*ts)[i].Features = append([]float64{1}, (*ts)[i].Features...)
	}
	return nil
}

// getTargetVector returns the target vector from the training set.
func (ts *TrainingSet) getTargetVector() []float64 {
	y := make([]float64, len(*ts))
	for i, te := range *ts {
		y[i] = te.Target
	}
	return y
}

// getDesignMatrix returns the design matrix from the training set.
func (ts *TrainingSet) getDesignMatrix() [][]float64 {
	d := make([][]float64, len(*ts))
	for i, te := range *ts {
		d[i] = te.Features
	}
	return d
}
