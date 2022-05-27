package ts

import (
	"reflect"
	"testing"

	"github.com/erni27/regression"
)

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

func TestAddDummies(t *testing.T) {
	tests := []struct {
		name string
		x    [][]float64
		want [][]float64
	}{
		{
			name: "n=1 m=2",
			x: [][]float64{
				{2},
				{5},
			},
			want: [][]float64{
				{1, 2},
				{1, 5},
			},
		},
		{
			name: "n=3 m=5",
			x: [][]float64{
				{17, -2, 15},
				{7, 2, 3},
				{12, 7, -8},
				{1, 89, -3},
				{5, 4, 3},
			},
			want: [][]float64{
				{1, 17, -2, 15},
				{1, 7, 2, 3},
				{1, 12, 7, -8},
				{1, 1, 89, -3},
				{1, 5, 4, 3},
			}},
		{
			name: "n=6 m=1",
			x: [][]float64{
				{1, 22.6, 18, 2, 15, -8},
			},
			want: [][]float64{
				{1, 1, 22.6, 18, 2, 15, -8},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AddDummies(tt.x)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name string
		ts   regression.TrainingSet
		want error
	}{
		{
			name: "valid",
			ts: regression.TrainingSet{
				X: [][]float64{
					{12, 5, 8, 9},
					{11, 8, 15, 7},
					{6, 10, 18, 6},
					{4, 15, 33, 5},
					{2, 22, 48, 1},
				},
				Y: []float64{11, 21, 43, 44, 15},
			},
			want: nil,
		},
		{
			name: "irregular matrix",
			ts: regression.TrainingSet{
				X: [][]float64{
					{12, 5, 8, 9},
					{11, 8, 15},
					{6, 10, 18, 6},
					{4, 15, 33, 5},
					{2, 22, 48, 1},
				},
				Y: []float64{11, 21, 43, 44, 15},
			},
			want: regression.ErrInvalidTrainingSet,
		},
		{
			name: "number of features greater than number of training examples",
			ts: regression.TrainingSet{
				X: [][]float64{
					{12, 5, 8, 9},
					{11, 8, 15, 7},
					{6, 10, 18, 6},
				},
				Y: []float64{11, 21, 43, 44, 15},
			},
			want: regression.ErrInvalidTrainingSet,
		},
		{
			name: "target vector length does not equal number of rows in a design matrix",
			ts: regression.TrainingSet{
				X: [][]float64{
					{12, 5, 8, 9},
					{11, 8, 15, 7},
					{6, 10, 18, 6},
					{4, 15, 33, 5},
					{2, 22, 48, 1},
				},
				Y: []float64{11, 21, 43, 44},
			},
			want: nil,
		},
		{
			name: "zero value training set",
			ts:   regression.TrainingSet{},
			want: regression.ErrInvalidTrainingSet,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Validate(tt.ts); err != tt.want {
				t.Fatalf("got %v, want %v", err, tt.want)
			}
		})
	}
}
