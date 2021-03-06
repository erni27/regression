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

func LoadTrainingSet(path string) (regression.TrainingSet, error) {
	f, err := os.Open("./testdata/" + path)
	if err != nil {
		return regression.TrainingSet{}, err
	}
	defer f.Close()
	data, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return regression.TrainingSet{}, err
	}
	m := len(data)
	x := make([][]float64, m)
	y := make([]float64, m)
	for j, l := range data {
		x[j] = make([]float64, len(l)-1)
		for i, f := range l[:len(l)-1] {
			f, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return regression.TrainingSet{}, err
			}
			x[j][i] = f
		}
		f, err := strconv.ParseFloat(l[len(l)-1], 64)
		if err != nil {
			return regression.TrainingSet{}, err
		}
		y[j] = f
	}
	return regression.TrainingSet{X: x, Y: y}, nil
}
