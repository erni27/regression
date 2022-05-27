// Package scaling contains implementation of feature scaling.
// It supports two types of feature scaling: normalization and standarization.
package scaling

import (
	"errors"
	"math"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/matrix"
)

var (
	// ErrUnsupportedTechnique is returned if an unsupported feature scaling technique was chosen.
	ErrUnsupportedTechnique = errors.New("unsupported scaling technique")
	// ErrInvalidParameters is returned if scaling parameters vectors have different length.
	ErrInvalidParameters = errors.New("invalid scaling parameters")
)

// Technique identifies a feature scaling technique.
type Technique int

const (
	Normalization = iota + 1
	Standarization
)

// Result holds the scaled features set along with the scaling parameters.
type Result struct {
	X          [][]float64
	Parameters Parameters
}

// Parameters group together two parameters used in scaling.
type Parameters struct {
	U []float64 // mean value of all features from a design matrix
	S []float64 // range or standard deviation of all features from a design matrix
}

// Scale scales a single feature vector with given parameters.
func Scale(v []float64, p Parameters) ([]float64, error) {
	if len(p.U) != len(p.S) {
		return nil, ErrInvalidParameters
	}
	if len(v) != len(p.U) {
		return nil, regression.ErrInvalidFeatureVector
	}
	n := len(v)
	sc := make([]float64, n)
	for i := 0; i < n; i++ {
		sc[i] = (v[i] - p.U[i]) / p.S[i]
	}
	return sc, nil
}

// ScaleDesignMatrix scales a design matrix with a given technique.
func ScaleDesignMatrix(t Technique, x [][]float64) (Result, error) {
	if !matrix.IsRegular(x) {
		return Result{}, regression.ErrInvalidDesignMatrix
	}
	switch t {
	case Normalization:
		return normalize(x)
	case Standarization:
		return standarize(x)
	default:
		return Result{}, ErrUnsupportedTechnique
	}
}

// normalize performs feature scaling through normalization.
func normalize(x [][]float64) (Result, error) {
	m := len(x)
	n := len(x[0])
	min := make([]float64, n)
	max := make([]float64, n)
	for i := 0; i < n; i++ {
		min[i] = math.Inf(1)
		max[i] = math.Inf(-1)
	}
	mean := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			mean[j] += x[i][j]
			if max[j] < x[i][j] {
				max[j] = x[i][j]
			}
			if min[j] > x[i][j] {
				min[j] = x[i][j]
			}
		}
	}
	ran := make([]float64, n)
	for i := 0; i < n; i++ {
		mean[i] /= float64(m)
		ran[i] = max[i] - min[i]
		if ran[i] == 0 {
			return Result{}, regression.ErrInvalidDesignMatrix
		}
	}
	p := Parameters{U: mean, S: ran}
	sm, err := scaleDesignMatrix(x, p)
	if err != nil {
		return Result{}, err
	}
	return Result{X: sm, Parameters: p}, nil
}

// standarize performs feature scaling through standarization.
func standarize(x [][]float64) (Result, error) {
	m := len(x)
	n := len(x[0])
	mean := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			mean[j] += x[i][j]
		}
	}
	for i := 0; i < n; i++ {
		mean[i] /= float64(m)
	}
	dev := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dev[j] += math.Pow(x[i][j]-mean[j], 2)
			if dev[j] == 0 {
				return Result{}, regression.ErrInvalidDesignMatrix
			}
		}
	}
	for i := 0; i < n; i++ {
		dev[i] /= float64(m)
		dev[i] = math.Sqrt(dev[i])
	}
	p := Parameters{U: mean, S: dev}
	sm, err := scaleDesignMatrix(x, p)
	if err != nil {
		return Result{}, err
	}
	return Result{X: sm, Parameters: p}, nil
}

// scaleDesignMatrix scales a design matrix with given scaling parameters.
func scaleDesignMatrix(x [][]float64, p Parameters) ([][]float64, error) {
	sm := make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		sv, err := Scale(x[i], p)
		if err != nil {
			return nil, err
		}
		sm[i] = append(sm[i], sv...)
	}
	return sm, nil
}
