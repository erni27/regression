package regression

import (
	"testing"

	"github.com/erni27/regression/internal/assert"
)

func TestTrain(t *testing.T) {
	type expected struct {
		r2     float64
		coeffs []float64
	}
	tests := []struct {
		name     string
		fileName string
		want     expected
	}{
		{
			name:     "n=1,m=97",
			fileName: "n=1_m=97.txt",
			want:     expected{r2: 0.702, coeffs: []float64{-3.896, 1.193}},
		},
		{
			name:     "n=2,m=47",
			fileName: "n=2_m=47.txt",
			want:     expected{r2: 0.733, coeffs: []float64{89597.91, 139.211, -8738.019}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts, err := loadTrainingSet(tt.fileName)
			if err != nil {
				t.Fatalf("cannot load training set %v", err)
			}
			got, err := Train(ts)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			coeffs, err := got.Coefficients()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !assert.AreFloatSlicesEqual(coeffs, tt.want.coeffs, 3) {
				t.Errorf("got coefficients %v, want %v", coeffs, tt.want.coeffs)
			}
			r2, err := got.R2()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !assert.AreFloatEqual(r2, tt.want.r2, 2) {
				t.Errorf("got r2 %v, want %v", r2, tt.want.r2)
			}
		})
	}
}
