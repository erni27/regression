package assert

import (
	"math"

	"golang.org/x/exp/constraints"
)

func AreFloatEqual[float constraints.Float](x, y float, precision uint) bool {
	return round(x, precision) == round(y, precision)
}

func AreFloatSlicesEqual[float constraints.Float](x, y []float, precision int) bool {
	if len(x) != len(y) {
		return false
	}
	var equal func(float, float) bool
	if precision < 0 {
		equal = func(a, b float) bool { return a == b }
	} else {
		equal = func(a, b float) bool { return AreFloatEqual(a, b, uint(precision)) }
	}
	for i := 0; i < len(x); i++ {
		if !equal(x[i], y[i]) {
			return false
		}
	}
	return true
}

func Are2DFloatSlicesEqual[float constraints.Float](x, y [][]float, precision int) bool {
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

func round[float constraints.Float](v float, p uint) float {
	r := math.Pow(10, float64(p))
	return float(math.Round(float64(v)*r) / r)
}
