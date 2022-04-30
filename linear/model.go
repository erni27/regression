package linear

import (
	"errors"
	"fmt"
	"math"

	"github.com/erni27/regression"
)

var (
	ErrInvalidFeatureVector = errors.New("invalid feature vector")
	ErrNotTrainedModel      = errors.New("not trained model")
	ErrInvalidTrainingSet   = errors.New("invalid training set")
)

// A Model is a linear regression model.
type Model struct {
	coeffs []float64
	r2     float64
}

// Predict returns the predicated target value for the given input.
func (m Model) Predict(x []float64) (float64, error) {
	if !m.IsTrained() {
		return 0, ErrNotTrainedModel
	}
	return calcHypho(x, m.coeffs)
}

// Coefficients returns the trained linear regression model's coefficients.
func (m Model) Coefficients() ([]float64, error) {
	if !m.IsTrained() {
		return nil, ErrNotTrainedModel
	}
	return m.coeffs, nil
}

// R2 returns 'R squared'.
func (m Model) Accuracy() (float64, error) {
	if !m.IsTrained() {
		return 0, ErrNotTrainedModel
	}
	return m.r2, nil
}

func (m Model) String() string {
	if !m.IsTrained() {
		return ErrNotTrainedModel.Error()
	}
	s := fmt.Sprintf("y = %f", m.coeffs[0])
	for i, coeff := range m.coeffs[1:] {
		s += fmt.Sprintf(" + x%d*%f", i+1, coeff)
	}
	return s
}

// IsTrained checks if linear regression model is trained.
func (m Model) IsTrained() bool {
	return m.coeffs != nil
}

// calcHypho calculates the hyphothesis function.
//
// The hyphothesis equals h(x)=OX, where O stands for a coefficients vector and X is a feature vector.
// It includes dummy feature during the calculation.
func calcHypho(x []float64, coeffs []float64) (float64, error) {
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

func calcMean(y []float64) float64 {
	var s float64
	for _, v := range y {
		s += v
	}
	return s / float64(len(y))
}

// addDummyFeatures puts dummy feature (equals 1) for each training example being a part of the training set.
//
// If addDummyFeatures finds inconsistency in the training set, it also returns an error.
func addDummyFeatures(ts *regression.TrainingSet[float64]) error {
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
func getTargetVector(ts regression.TrainingSet[float64]) []float64 {
	y := make([]float64, len(ts))
	for i, te := range ts {
		y[i] = te.Target
	}
	return y
}

// getDesignMatrix returns the design matrix from the training set.
func getDesignMatrix(ts regression.TrainingSet[float64]) [][]float64 {
	d := make([][]float64, len(ts))
	for i, te := range ts {
		d[i] = te.Features
	}
	return d
}

// calcR2 calculates the coefficient of determination (R squared).
func calcR2(ts regression.TrainingSet[float64], coeffs []float64) (float64, error) {
	var ssr, sst float64
	m := calcMean(getTargetVector(ts))
	for _, te := range ts {
		// calcHypho includes dummy feature by itself.
		v, err := calcHypho(te.Features[1:], coeffs)
		if err != nil {
			return 0, err
		}
		ssr += math.Pow(te.Target-v, 2)
		sst += math.Pow(te.Target-m, 2)
	}
	return 1 - ssr/sst, nil
}
