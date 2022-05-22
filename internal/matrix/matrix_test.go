package matrix

import (
	"context"
	"reflect"
	"testing"

	"github.com/erni27/regression/internal/regressiontest"
)

func TestInverse(t *testing.T) {
	tests := []struct {
		name   string
		matrix [][]float64
		want   [][]float64
	}{
		{
			name: "4x4 without zeros on main diagonal",
			matrix: [][]float64{
				{3, 3, -4, -3},
				{0, 6, 1, 1},
				{5, 4, 2, 1},
				{2, 3, 3, 2},
			},
			want: [][]float64{
				{-7, 5, 12, -19},
				{3, -2, -5, 8},
				{41, -30, -69, 111},
				{-59, 43, 99, -159},
			},
		},
		{
			name: "4x4 with zeros on main diagonal",
			matrix: [][]float64{
				{0, 1, -1, 1},
				{2, 2, 0, -2},
				{1, 1, -2, 0},
				{0, 1, 2, 0},
			},
			want: [][]float64{
				{-4, -2, 5, 3},
				{2, 1, -2, -1},
				{-1, -0.5, 1, 1},
				{-2, -1.5, 3, 2},
			},
		},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Inverse(ctx, tt.matrix)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !regressiontest.Are2DFloatSlicesEqual(got, tt.want, 6) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMultiply(t *testing.T) {
	tests := []struct {
		name string
		x    [][]float64
		y    [][]float64
		want [][]float64
	}{
		{
			name: "3x4 x 3x3",
			x: [][]float64{
				{1, 0, 1},
				{2, 1, 1},
				{0, 1, 1},
				{1, 1, 2},
			},
			y: [][]float64{
				{1, 2, 1},
				{2, 3, 1},
				{4, 2, 2},
			},
			want: [][]float64{
				{5, 4, 3},
				{8, 9, 5},
				{6, 5, 3},
				{11, 9, 6},
			},
		},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Multiply(ctx, tt.x, tt.y)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !regressiontest.Are2DFloatSlicesEqual(got, tt.want, -1) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	tests := []struct {
		name   string
		matrix [][]float64
		want   [][]float64
	}{
		{
			name: "4x4",
			matrix: [][]float64{
				{3, 3, -4, -3},
				{0, 6, 1, 1},
				{5, 4, 2, 1},
				{2, 3, 3, 2},
			},
			want: [][]float64{
				{3, 0, 5, 2},
				{3, 6, 4, 3},
				{-4, 1, 2, 3},
				{-3, 1, 1, 2},
			},
		},
		{
			name: "3x4",
			matrix: [][]float64{
				{0, 1, -1, 1},
				{2, 2, 0, -2},
				{1, 1, -2, 0},
			},
			want: [][]float64{
				{0, 2, 1},
				{1, 2, 1},
				{-1, 0, -2},
				{1, -2, 0},
			},
		},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Transpose(ctx, tt.matrix)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !regressiontest.Are2DFloatSlicesEqual(got, tt.want, -1) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMultiplyByVector(t *testing.T) {
	tests := []struct {
		name string
		x    [][]float64
		y    []float64
		want []float64
	}{
		{
			name: "2x3",
			x: [][]float64{
				{1, -1, 2},
				{0, -3, 1},
			},
			y:    []float64{2, 1, 0},
			want: []float64{1, -3},
		},
	}
	ctx := context.Background()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MultiplyByVector(ctx, tt.x, tt.y)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAddDummy(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		want []float64
	}{
		{
			name: "n=1",
			x:    []float64{2},
			want: []float64{1, 2},
		},
		{
			name: "n=3",
			x:    []float64{17, -2, 15},
			want: []float64{1, 17, -2, 15},
		},
		{
			name: "n=6",
			x:    []float64{1, 22.6, 18, 2, 15, -8},
			want: []float64{1, 1, 22.6, 18, 2, 15, -8},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AddDummy(tt.x)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}
