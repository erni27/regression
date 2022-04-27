package regression

import (
	"reflect"
	"testing"
)

func TestPredict(t *testing.T) {
	tests := []struct {
		name   string
		arg    []float64
		coeffs []float64
		want   float64
	}{
		{name: "n=1", arg: []float64{5}, coeffs: []float64{17, 0.2}, want: 18},
		{name: "n=2", arg: []float64{1.2, 21}, coeffs: []float64{997, 5, 0.5}, want: 1013.5},
		{name: "n=3", arg: []float64{0.1, 2, 6}, coeffs: []float64{2.73, 10, 0.5, 1}, want: 10.73},
		{name: "n=4", arg: []float64{112, 3, 14, 0}, coeffs: []float64{5, 1, 2, 0.5, 0.3333}, want: 130},
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
				t.Fatalf("want %f, got %f", tt.want, got)
			}
		})
	}
}

func TestPredictInvalidFeatureVector(t *testing.T) {
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
			if err != ErrInvalidFeatureVector {
				t.Fatalf("want %v, got %v", ErrInvalidFeatureVector, err)
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
			got, err := m.Coefficients()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !reflect.DeepEqual(got, m.coeffs) {
				t.Fatalf("want %v, got %v", got, tt.coeffs)
			}
		})
	}
}

func TestR2(t *testing.T) {
	coeffs := []float64{1, 2}
	tests := []struct {
		name string
		r2   float64
	}{
		{name: "r2=0.8", r2: 0.8},
		{name: "r2=0.75", r2: 0.75},
		{name: "r2=0.77", r2: 0.77},
		{name: "r2=0.81", r2: 0.81},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := model{
				coeffs: coeffs,
				r2:     tt.r2,
			}
			got, err := m.R2()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if got != m.r2 {
				t.Fatalf("want %f, got %f", got, m.r2)
			}
		})
	}
}

func TestNotTrainedModel(t *testing.T) {
	m := model{}
	t.Run("Predict()", func(t *testing.T) {
		_, err := m.Predict([]float64{1, 2, 3})
		if err != ErrNotTrainedModel {
			t.Fatalf("Predict(want %v, got %v", ErrNotTrainedModel, err)
		}
	})
	t.Run("Coefficients()", func(t *testing.T) {
		_, err := m.Coefficients()
		if err != ErrNotTrainedModel {
			t.Errorf("Predict(want %v, got %v", ErrNotTrainedModel, err)
		}
	})
	t.Run("R2()", func(t *testing.T) {
		_, err := m.R2()
		if err != ErrNotTrainedModel {
			t.Errorf("Predict(want %v, got %v", ErrNotTrainedModel, err)
		}
	})
	t.Run("String()", func(t *testing.T) {
		got := m.String()
		if got != ErrNotTrainedModel.Error() {
			t.Errorf("Predict(want %s, got %s", ErrNotTrainedModel.Error(), got)
		}
	})
}
