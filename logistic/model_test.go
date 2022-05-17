package logistic

import (
	"reflect"
	"testing"

	"github.com/erni27/regression"
)

func TestPredict(t *testing.T) {
	tests := []struct {
		name   string
		arg    []float64
		coeffs []float64
		want   int
	}{
		{name: "n=1", arg: []float64{5}, coeffs: []float64{17, 0.2}, want: 1},
		{name: "n=2", arg: []float64{1.2, 21}, coeffs: []float64{-997, 5, 0.5}, want: 0},
		{name: "n=3", arg: []float64{0.1, 2, 6}, coeffs: []float64{2.73, 10, 0.5, 1}, want: 1},
		{name: "n=4", arg: []float64{12, 3, 14, 0}, coeffs: []float64{-55, 1, 2, 0.5, 0.3333}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := model{
				coeffs: tt.coeffs,
			}
			got, err := m.Predict(tt.arg)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if got != tt.want {
				t.Fatalf("want %d, got %d", tt.want, got)
			}
		})
	}
}

func TestPredict_InvalidFeatureVector(t *testing.T) {
	tests := []struct {
		name   string
		arg    []float64
		coeffs []float64
		want   float64
	}{
		{name: "n=1", arg: []float64{}, coeffs: []float64{17, 0.2}, want: 18},
		{name: "n=2", arg: []float64{1.2, 21, 15}, coeffs: []float64{997, 5, 0.5}, want: 1013.5},
		{name: "n=3", arg: []float64{0.1}, coeffs: []float64{2.73, 10, 0.5, 1}, want: 10.73},
		{name: "n=4", arg: []float64{112, 3, 14}, coeffs: []float64{5, 1, 2, 0.5, 0.3333}, want: 130},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := model{
				coeffs: tt.coeffs,
			}
			_, err := m.Predict(tt.arg)
			if err != regression.ErrInvalidFeatureVector {
				t.Fatalf("want %v, got %v", regression.ErrInvalidFeatureVector, err)
			}
		})
	}
}

func TestCoefficients(t *testing.T) {
	tests := []struct {
		name   string
		coeffs []float64
	}{
		{name: "n=1", coeffs: []float64{17, 0.2}},
		{name: "n=2", coeffs: []float64{997, 5, 0.5}},
		{name: "n=3", coeffs: []float64{2.73, 10, 0.5, 1}},
		{name: "n=4", coeffs: []float64{5, 1, 2, 0.5, 0.3333}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := model{
				coeffs: tt.coeffs,
			}
			got := m.Coefficients()
			if !reflect.DeepEqual(got, m.coeffs) {
				t.Fatalf("want %v, got %v", got, tt.coeffs)
			}
		})
	}
}

func TestAccuracy(t *testing.T) {
	coeffs := []float64{1, 2}
	tests := []struct {
		name string
		acc  float64
	}{
		{name: "r2=0.8", acc: 0.8},
		{name: "r2=0.75", acc: 0.75},
		{name: "r2=0.77", acc: 0.77},
		{name: "r2=0.81", acc: 0.81},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := model{
				coeffs: coeffs,
				acc:    tt.acc,
			}
			got := m.Accuracy()
			if got != m.acc {
				t.Fatalf("want %f, got %f", got, m.acc)
			}
		})
	}
}
