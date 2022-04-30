package regressiontest

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"

	"github.com/erni27/regression"
)

func AreFloatEqual(x, y float64, precision uint) bool {
	return Round(x, precision) == Round(y, precision)
}

func AreFloatSlicesEqual(x, y []float64, precision int) bool {
	if len(x) != len(y) {
		return false
	}
	var equal func(float64, float64) bool
	if precision < 0 {
		equal = func(a, b float64) bool { return a == b }
	} else {
		equal = func(a, b float64) bool { return AreFloatEqual(a, b, uint(precision)) }
	}
	for i := 0; i < len(x); i++ {
		if !equal(x[i], y[i]) {
			return false
		}
	}
	return true
}

func Are2DFloatSlicesEqual(x, y [][]float64, precision int) bool {
	if len(x) != len(y) {
		return false
	}
	for i := 0; i < len(x); i++ {
		if !AreFloatSlicesEqual(x[i], y[i], precision) {
			return false
		}
	}
	return true
}

func Round(v float64, p uint) float64 {
	r := math.Pow(10, float64(p))
	return math.Round(v*r) / r
}

func LoadTrainingSet[T regression.TargetType](fn string) (regression.TrainingSet[T], error) {
	f, err := os.Open("../testdata/" + fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	var ts regression.TrainingSet[T] = make([]regression.TrainingExample[T], len(data))
	for j, l := range data {
		te := regression.TrainingExample[T]{Features: make([]float64, len(l)-1)}
		for i, f := range l[:len(l)-1] {
			f, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return nil, err
			}
			te.Features[i] = f
		}
		f, err := strconv.ParseFloat(l[len(l)-1], 64)
		if err != nil {
			return nil, err
		}
		te.Target = T(f)
		ts[j] = te
	}
	return ts, nil
}
