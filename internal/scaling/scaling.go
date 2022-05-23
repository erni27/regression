package scaling

import (
	"math"

	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

// FeatureScaler is the interface that wraps the basic Scale method.
type FeatureScaler interface {
	// Scale scales a design matrix.
	Scale([][]float64) (Result, error)
}

// ScalerFunc is an adapter to allow the use of plain functions as scalers.
type ScalerFunc func([][]float64) (Result, error)

func (f ScalerFunc) Scale(x [][]float64) (Result, error) {
	return f(x)
}

// Scale scales a single feature vector with given scaling parameters.
func Scale(x, u, s []float64) ([]float64, error) {
	if len(x) != len(u) || len(u) != len(s) {
		return nil, regression.ErrInvalidFeatureVector
	}
	n := len(x)
	sc := make([]float64, n)
	for i := 0; i < n; i++ {
		sc[i] = (x[i] - u[i]) / s[i]
	}
	return sc, nil
}

// NewScaler returns a new scaler. If unsupported FeatureScalingTechnique is passed, an error is returned.
func NewScaler(fst options.FeatureScalingTechnique) (FeatureScaler, error) {
	var s ScalerFunc
	switch fst {
	case options.None:
		s = none
	case options.Normalization:
		s = normalize
	case options.Standarization:
		s = standarize
	default:
		return nil, regression.ErrUnsupportedScalingTechnique
	}
	return s, nil
}

// Result holds the scaled features set along with the scaling parameters.
type Result struct {
	X [][]float64
	U []float64
	S []float64
}

// normalize performs feature scaling through normalization.
func normalize(x [][]float64) (Result, error) {
	m := len(x)
	n := len(x[0]) - 1
	min := make([]float64, n)
	max := make([]float64, n)
	for i := 0; i < n; i++ {
		min[i] = math.Inf(1)
		max[i] = math.Inf(-1)
	}
	mean := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			mean[j] += x[i][j+1]
			if max[j] < x[i][j+1] {
				max[j] = x[i][j+1]
			}
			if min[j] > x[i][j+1] {
				min[j] = x[i][j+1]
			}
		}
	}
	ran := make([]float64, n)
	for i := 0; i < n; i++ {
		mean[i] /= float64(m)
		ran[i] = max[i] - min[i]
	}
	sm, err := scaleDesignMatrix(x, mean, ran)
	if err != nil {
		return Result{}, err
	}
	return Result{X: sm, U: mean, S: ran}, nil
}

// standarize performs feature scaling through standarization.
func standarize(x [][]float64) (Result, error) {
	m := len(x)
	n := len(x[0]) - 1
	mean := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			mean[j] += x[i][j+1]
		}
	}
	for i := 0; i < n; i++ {
		mean[i] /= float64(m)
	}
	dev := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dev[j] += math.Pow(x[i][j+1]-mean[j], 2)
		}
	}
	for i := 0; i < n; i++ {
		dev[i] /= float64(m)
		dev[i] = math.Sqrt(dev[i])
	}
	sm, err := scaleDesignMatrix(x, mean, dev)
	if err != nil {
		return Result{}, err
	}
	return Result{X: sm, U: mean, S: dev}, nil
}

// none does not perform feature scaling.
// It returns an unchanged design matrix along with the identity scaling parameters.
// A feature vector scaled with those parameters remains the same.
func none(x [][]float64) (Result, error) {
	n := len(x[0]) - 1
	u := make([]float64, n)
	s := make([]float64, n)
	for i := 0; i < n; i++ {
		s[i] = 1
	}
	return Result{X: x, U: u, S: s}, nil
}

// scaleDesignMatrix scales a design matrix with given scaling parameters.
func scaleDesignMatrix(x [][]float64, u, s []float64) ([][]float64, error) {
	sm := make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		sm[i] = make([]float64, 1, len(u))
		sm[i][0] = 1
		sv, err := Scale(x[i][1:], u, s)
		if err != nil {
			return nil, err
		}
		sm[i] = append(sm[i], sv...)
	}
	return sm, nil
}
