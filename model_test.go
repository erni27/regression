package regression

import (
	"testing"
)

func TestPredict(t *testing.T) {
	type fields struct {
		learningRate float64
		coefficients Vector
		r2           float64
	}
	type args struct {
		x Vector
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{}
	m := model{
		coefficients: []float64{0.5, 1.8, 13, 12, 5.3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := m.Predict(tt.args.x)
			if (err != nil) != tt.wantErr {
				t.Errorf("model.Predict() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("model.Predict() = %v, want %v", got, tt.want)
			}
		})
	}
}
