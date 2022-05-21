package gd

import (
	"math"

	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

// FeatureScaler is the interface that wraps the basic Scale method.
type FeatureScaler interface {
	Scale([][]float64) ScalingResult
}

// ScalerFunc is an adapter to allow the use of plain functions as scalers.
type ScalerFunc func([][]float64) ScalingResult

func (f ScalerFunc) Scale(x [][]float64) ScalingResult {
	return f(x)
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

// ScalingResult holds the scaled features set along with the scaling parameters.
type ScalingResult struct {
	X [][]float64
	U []float64
	S []float64
}

// normalize performs features scaling through normalization.
func normalize(x [][]float64) ScalingResult {
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
	norm := make([][]float64, m)
	for i := 0; i < m; i++ {
		norm[i] = make([]float64, n+1)
		norm[i][0] = 1
		for j := 0; j < n; j++ {
			norm[i][j+1] = (x[i][j+1] - mean[j]) / ran[j]
		}
	}
	return ScalingResult{X: norm, U: mean, S: ran}
}

// standarize performs features scaling through standarization.
func standarize(x [][]float64) ScalingResult {
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
	v := make([]float64, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			v[j] += math.Pow(x[i][j+1]-mean[j], 2)
		}
	}
	for i := 0; i < n; i++ {
		v[i] /= float64(m)
		v[i] = math.Sqrt(v[i])
	}
	stand := make([][]float64, m)
	for i := 0; i < m; i++ {
		stand[i] = make([]float64, n+1)
		stand[i][0] = 1
		for j := 0; j < n; j++ {
			stand[i][j+1] = (x[i][j+1] - mean[j]) / v[j]
		}
	}
	return ScalingResult{X: stand, U: mean, S: v}
}

// none does not perform feature scaling.
// It returns unchanged features set along with the identity scaling parameters.
// A features vector scaled with those parameters remains the same.
func none(x [][]float64) ScalingResult {
	n := len(x[0]) - 1
	u := make([]float64, n)
	s := make([]float64, n)
	for i := 0; i < n; i++ {
		s[i] = 1
	}
	return ScalingResult{X: x, U: u, S: s}
}
