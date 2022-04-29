package regression

import (
	"errors"

	"golang.org/x/exp/constraints"
)

var (
	ErrInvalidTrainingSet = errors.New("invalid training set")
)

// TrainingExample represents a single (features, target) example.
type TrainingExample[float constraints.Float] struct {
	Features []float
	Target   float
}

// TrainingSet represents set of traning examples.
type TrainingSet[float constraints.Float] []TrainingExample[float]

// addDummyFeatures puts dummy feature (equals 1) for each training example being a part of the training set.
//
// If addDummyFeatures finds inconsistency in the training set, it also returns an error.
func (ts *TrainingSet[float]) addDummyFeatures() error {
	// n represents the number of features
	n := len((*ts)[0].Features)
	for i := 0; i < len(*ts); i++ {
		if n != len((*ts)[i].Features) {
			return ErrInvalidTrainingSet
		}
		(*ts)[i].Features = append([]float{1}, (*ts)[i].Features...)
	}
	return nil
}

// getTargetVector returns the target vector from the training set.
func (ts *TrainingSet[float]) getTargetVector() []float {
	y := make([]float, len(*ts))
	for i, te := range *ts {
		y[i] = te.Target
	}
	return y
}

// getDesignMatrix returns the design matrix from the training set.
func (ts *TrainingSet[float]) getDesignMatrix() [][]float {
	d := make([][]float, len(*ts))
	for i, te := range *ts {
		d[i] = te.Features
	}
	return d
}

// calcR2 calculates the coefficient of determination (R squared).
func calcR2[float constraints.Float](ts TrainingSet[float], coeffs []float) (float, error) {
	var ssr, sst float
	m := calcMean(ts.getTargetVector())
	for _, te := range ts {
		// calcHypho includes dummy feature by itself.
		v, err := calcHypho(te.Features[1:], coeffs)
		if err != nil {
			return 0, err
		}
		ssr += (te.Target - v) * (te.Target - v)
		sst += (te.Target - m) * (te.Target - m)
	}
	return 1 - ssr/sst, nil
}
