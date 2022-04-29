package assert

import "math"

func AreFloatEqual(x, y float64, precision uint) bool {
	return round(x, precision) == round(y, precision)
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

func round(v float64, p uint) float64 {
	r := math.Pow(10, float64(p))
	return math.Round(v*r) / r
}
