package gd

import (
	"math"

	"github.com/erni27/regression"
	"github.com/erni27/regression/options"
)

type FeatureScaler interface {
	Scale([][]float64) ScalingResult
}

type ScalerFunc func([][]float64) ScalingResult

func (f ScalerFunc) Scale(x [][]float64) ScalingResult {
	return f(x)
}

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

type ScalingResult struct {
	X [][]float64
	U []float64
	S []float64
}

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
			mean[j] += x[i][j]
			if max[j] < x[i][j] {
				max[j] = x[i][j]
			} else if min[j] > x[i][j] {
				min[j] = x[i][j]
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
		norm[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			norm[i][j] = (x[i][j] - mean[j]) / ran[j]
		}
	}
	return ScalingResult{X: norm, U: mean, S: ran}
}

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
	}
	stand := make([][]float64, m)
	for i := 0; i < m; i++ {
		stand[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			stand[i][j] = (x[i][j+1] - mean[j]) / v[j]
		}
	}
	return ScalingResult{X: stand, U: mean, S: v}
}

func none(x [][]float64) ScalingResult {
	n := len(x[0]) - 1
	u := make([]float64, n)
	s := make([]float64, n)
	for i := 0; i < n; i++ {
		s[i] = 1
	}
	return ScalingResult{X: x, U: u, S: s}
}
