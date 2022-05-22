package scaling

import (
	"testing"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/regressiontest"
	"github.com/erni27/regression/options"
)

func TestNewScaler(t *testing.T) {
	tests := []struct {
		name    string
		fts     options.FeatureScalingTechnique
		wantErr bool
		err     error
	}{
		{name: "normalization", fts: options.Normalization, wantErr: false},
		{name: "standarization", fts: options.Standarization, wantErr: false},
		{name: "none", fts: options.None, wantErr: false},
		{name: "unsupported", fts: -1, wantErr: true, err: regression.ErrUnsupportedScalingTechnique},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewScaler(tt.fts)
			if (err != nil) != tt.wantErr {
				if err != tt.err {
					t.Fatalf("want error %v, got %v", tt.err, err)
				}
			}
			if !tt.wantErr && got == nil {
				t.Fatalf("want not nil scaler")
			}
		})
	}
}

func TestScale(t *testing.T) {
	tests := []struct {
		name string
		fts  options.FeatureScalingTechnique
		x    [][]float64
		want Result
	}{
		{
			name: "standarization n=1 m=8",
			fts:  options.Standarization,
			x:    [][]float64{{1, 2}, {1, 4}, {1, 4}, {1, 4}, {1, 5}, {1, 5}, {1, 7}, {1, 9}},
			want: Result{
				X: [][]float64{{1, -1.5}, {1, -0.5}, {1, -0.5}, {1, -0.5}, {1, 0}, {1, 0}, {1, 1}, {1, 2}},
				U: []float64{5},
				S: []float64{2},
			},
		},
		{
			name: "normalization n=1 m=8",
			fts:  options.Normalization,
			x:    [][]float64{{1, 2}, {1, 4}, {1, 4}, {1, 4}, {1, 5}, {1, 5}, {1, 7}, {1, 9}},
			want: Result{
				X: [][]float64{{1, -0.429}, {1, -0.143}, {1, -0.143}, {1, -0.143}, {1, 0}, {1, 0}, {1, 0.286}, {1, 0.571}},
				U: []float64{5},
				S: []float64{7},
			},
		},
		{
			name: "none n=1 m=8",
			fts:  options.None,
			x:    [][]float64{{1, 2}, {1, 4}, {1, 4}, {1, 4}, {1, 5}, {1, 5}, {1, 7}, {1, 9}},
			want: Result{
				X: [][]float64{{1, 2}, {1, 4}, {1, 4}, {1, 4}, {1, 5}, {1, 5}, {1, 7}, {1, 9}},
				U: []float64{0},
				S: []float64{1},
			},
		},
		{
			name: "standarization n=2 m=10",
			fts:  options.Standarization,
			x:    [][]float64{{1, 2, 3}, {1, 4, 5}, {1, 4, 2}, {1, 4, 6}, {1, 5, 1}, {1, 5, 7}, {1, 7, 3}, {1, 9, 5}, {1, 2, 4}, {1, 8, 4}},
			want: Result{
				X: [][]float64{{1, -1.342, -0.577}, {1, -0.447, 0.577}, {1, -0.447, -1.155}, {1, -0.447, 1.155}, {1, 0, -1.732}, {1, 0, 1.732}, {1, 0.894, -0.577}, {1, 1.789, 0.577}, {1, -1.342, 0}, {1, 1.342, 0}},
				U: []float64{5, 4},
				S: []float64{2.236, 1.732},
			},
		},
		{
			name: "normalization n=2 m=10",
			fts:  options.Normalization,
			x:    [][]float64{{1, 2, 3}, {1, 4, 5}, {1, 4, 2}, {1, 4, 6}, {1, 5, 1}, {1, 5, 7}, {1, 7, 3}, {1, 9, 5}, {1, 2, 4}, {1, 8, 4}},
			want: Result{
				X: [][]float64{{1, -0.429, -0.167}, {1, -0.143, 0.167}, {1, -0.143, -0.333}, {1, -0.143, 0.333}, {1, 0, -0.5}, {1, 0, 0.5}, {1, 0.286, -0.167}, {1, 0.571, 0.167}, {1, -0.429, 0}, {1, 0.429, 0}},
				U: []float64{5, 4},
				S: []float64{7, 6},
			},
		},
		{
			name: "none n=2 m=10",
			fts:  options.None,
			x:    [][]float64{{1, 2, 3}, {1, 4, 5}, {1, 4, 2}, {1, 4, 6}, {1, 5, 1}, {1, 5, 7}, {1, 7, 3}, {1, 9, 5}, {1, 2, 4}, {1, 8, 4}},
			want: Result{
				X: [][]float64{{1, 2, 3}, {1, 4, 5}, {1, 4, 2}, {1, 4, 6}, {1, 5, 1}, {1, 5, 7}, {1, 7, 3}, {1, 9, 5}, {1, 2, 4}, {1, 8, 4}},
				U: []float64{0, 0},
				S: []float64{1, 1},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewScaler(tt.fts)
			if err != nil {
				t.Fatal(err)
			}
			got, err := s.Scale(tt.x)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !regressiontest.Are2DFloatSlicesEqual(got.X, tt.want.X, 3) {
				t.Errorf("want %v, got %v", tt.want.X, got.X)
			}
			if !regressiontest.AreFloatSlicesEqual(tt.want.U, got.U, 3) {
				t.Errorf("want %v, got %v", tt.want.U, got.U)
			}
			if !regressiontest.AreFloatSlicesEqual(tt.want.S, got.S, 3) {
				t.Errorf("want %v, got %v", tt.want.S, got.S)
			}
		})
	}
}
