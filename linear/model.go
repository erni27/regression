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
	// Include dummy feature equals 1 at the beginning.
	return hyphothesis(append([]float64{1}, x...), m.coeffs)
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

func calcMean(y []float64) float64 {
	var s float64
	for _, v := range y {
		s += v
	}
	return s / float64(len(y))
}

// calcR2 calculates the coefficient of determination (R squared).
func calcR2(s regression.TrainingSet, coeffs []float64) (float64, error) {
	var ssr, sst float64
	m := calcMean(s.GetTargetVector())
	for _, te := range s {
		v, err := hyphothesis(te.Features, coeffs)
		if err != nil {
			return 0, err
		}
		ssr += math.Pow(te.Target-v, 2)
		sst += math.Pow(te.Target-m, 2)
	}
	return 1 - ssr/sst, nil
}
