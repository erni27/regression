package scaling

import (
	"testing"

	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/regressiontest"
)

func TestScaleDesignMatrix(t *testing.T) {
	type want struct {
		x [][]float64
		u []float64
		s []float64
	}
	tests := []struct {
		name string
		t    Technique
		x    [][]float64
		want want
	}{
		{
			name: "standarization n=1 m=8",
			t:    Standarization,
			x:    [][]float64{{2}, {4}, {4}, {4}, {5}, {5}, {7}, {9}},
			want: want{
				x: [][]float64{{-1.5}, {-0.5}, {-0.5}, {-0.5}, {0}, {0}, {1}, {2}},
				u: []float64{5},
				s: []float64{2},
			},
		},
		{
			name: "normalization n=1 m=8",
			t:    Normalization,
			x:    [][]float64{{2}, {4}, {4}, {4}, {5}, {5}, {7}, {9}},
			want: want{
				x: [][]float64{{-0.429}, {-0.143}, {-0.143}, {-0.143}, {0}, {0}, {0.286}, {0.571}},
				u: []float64{5},
				s: []float64{7},
			},
		},
		{
			name: "standarization n=2 m=10",
			t:    Standarization,
			x:    [][]float64{{2, 3}, {4, 5}, {4, 2}, {4, 6}, {5, 1}, {5, 7}, {7, 3}, {9, 5}, {2, 4}, {8, 4}},
			want: want{
				x: [][]float64{{-1.342, -0.577}, {-0.447, 0.577}, {-0.447, -1.155}, {-0.447, 1.155}, {0, -1.732}, {0, 1.732}, {0.894, -0.577}, {1.789, 0.577}, {-1.342, 0}, {1.342, 0}},
				u: []float64{5, 4},
				s: []float64{2.236, 1.732},
			},
		},
		{
			name: "normalization n=2 m=10",
			t:    Normalization,
			x:    [][]float64{{2, 3}, {4, 5}, {4, 2}, {4, 6}, {5, 1}, {5, 7}, {7, 3}, {9, 5}, {2, 4}, {8, 4}},
			want: want{
				x: [][]float64{{-0.429, -0.167}, {-0.143, 0.167}, {-0.143, -0.333}, {-0.143, 0.333}, {0, -0.5}, {0, 0.5}, {0.286, -0.167}, {0.571, 0.167}, {-0.429, 0}, {0.429, 0}},
				u: []float64{5, 4},
				s: []float64{7, 6},
			},
		},
		{
			name: "normalization n=2 m=1",
			t:    Normalization,
			x:    [][]float64{{2, 3}, {4, 5}, {4, 2}, {4, 6}, {5, 1}, {5, 7}, {7, 3}, {9, 5}, {2, 4}, {8, 4}},
			want: want{
				x: [][]float64{{-0.429, -0.167}, {-0.143, 0.167}, {-0.143, -0.333}, {-0.143, 0.333}, {0, -0.5}, {0, 0.5}, {0.286, -0.167}, {0.571, 0.167}, {-0.429, 0}, {0.429, 0}},
				u: []float64{5, 4},
				s: []float64{7, 6},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ScaleDesignMatrix(tt.t, tt.x)
			if err != nil {
				t.Fatalf("want nil, got err %v", err)
			}
			if !regressiontest.Are2DFloatSlicesEqual(got.X, tt.want.x, 3) {
				t.Errorf("want %v, got %v", tt.want.x, got.X)
			}
			if !regressiontest.AreFloatSlicesEqual(tt.want.u, got.Parameters.U, 3) {
				t.Errorf("want %v, got %v", tt.want.u, got.Parameters.U)
			}
			if !regressiontest.AreFloatSlicesEqual(tt.want.s, got.Parameters.S, 3) {
				t.Errorf("want %v, got %v", tt.want.s, got.Parameters.S)
			}
		})
	}
}

func TestScaleDesignMatrix_Error(t *testing.T) {
	tests := []struct {
		name string
		t    Technique
		x    [][]float64
		want error
	}{
		{
			name: "standarize one row matrix",
			t:    Standarization,
			x: [][]float64{
				{2, 5},
			},
			want: regression.ErrInvalidDesignMatrix,
		},
		{
			name: "normalize one row matrix",
			t:    Normalization,
			x: [][]float64{
				{2, 5, 3},
			},
			want: regression.ErrInvalidDesignMatrix,
		},
		{
			name: "normalize nil matrix",
			t:    Normalization,
			want: regression.ErrInvalidDesignMatrix,
		},
		{
			name: "standarize irregular matrix",
			t:    Standarization,
			x: [][]float64{
				{5, 6, 2, 3},
				{3, 16, 87, 2},
				{22, 5, 8, 9, 7},
				{3, 12, 12, 5},
				{7, 8, -5, 4},
			},
			want: regression.ErrInvalidDesignMatrix,
		},
		{
			name: "unsupported scaling technique",
			t:    0,
			x: [][]float64{
				{5, 6, 2, 3},
				{3, 16, 87, 2},
				{22, 5, 8, 9},
				{3, 12, 12, 5},
				{7, 8, -5, 4},
			},
			want: ErrUnsupportedTechnique,
		},
		{
			name: "unsupported scaling technique",
			t:    0,
			x: [][]float64{
				{5, 6, 2, 3},
				{3, 16, 87, 2},
				{22, 5, 8, 9},
				{3, 12, 12, 5},
				{7, 8, -5, 4},
			},
			want: ErrUnsupportedTechnique,
		},
		{
			name: "normalize matrix with an unnecessary feature",
			t:    Normalization,
			x: [][]float64{
				{5, 6, 2, 3},
				{5, 16, 87, 2},
				{5, 5, 8, 9},
				{5, 12, 12, 5},
				{5, 8, -5, 4},
			},
			want: regression.ErrInvalidDesignMatrix,
		},
		{
			name: "standarize matrix with an unnecessary feature",
			t:    Standarization,
			x: [][]float64{
				{1, 6, 2, 2},
				{5, 16, 87, 2},
				{5, 5, 8, 2},
				{8, 12, 12, 2},
				{5, 8, -5, 2},
			},
			want: regression.ErrInvalidDesignMatrix,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ScaleDesignMatrix(tt.t, tt.x)
			if err != tt.want {
				t.Fatalf("want %v, got %v", tt.want, err)
			}
		})
	}
}

func TestScale(t *testing.T) {
	tests := []struct {
		name string
		v    []float64
		p    Parameters
		want []float64
	}{
		{
			name: "nil vector parameters and nil feature vector",
		},
		{
			name: "n=1",
			v:    []float64{15},
			p:    Parameters{U: []float64{10}, S: []float64{5}},
			want: []float64{1},
		},
		{
			name: "n=3",
			v:    []float64{2, 89, 8},
			p:    Parameters{U: []float64{1, 49, 7}, S: []float64{2, 80, 1}},
			want: []float64{0.5, 0.5, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Scale(tt.v, tt.p)
			if err != nil {
				t.Fatalf("want nil, got err %v", err)
			}
			if !regressiontest.AreFloatSlicesEqual(got, tt.want, 3) {
				t.Errorf("want %v, got %v", tt.want, got)
			}
		})
	}
}

func TestScale_Error(t *testing.T) {
	tests := []struct {
		name string
		v    []float64
		p    Parameters
		want error
	}{
		{
			name: "invalid feature vector",
			v:    []float64{17, 2, 8, -2},
			p:    Parameters{U: []float64{15, 1, 7}, S: []float64{2, 1, 2}},
			want: regression.ErrInvalidFeatureVector,
		},
		{
			name: "invalid parameters",
			v:    []float64{17, 2, 8, -2},
			p:    Parameters{U: []float64{15, 1, 7, 8}, S: []float64{2, 1, 2}},
			want: ErrInvalidParameters,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Scale(tt.v, tt.p)
			if err != tt.want {
				t.Fatalf("want %v, got %v", tt.want, err)
			}
		})
	}
}
