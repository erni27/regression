package regression

import (
	"reflect"
	"testing"
)

func TestNewTrainingSet(t *testing.T) {
	tests := []struct {
		name string
		exs  []TrainingExample
	}{
		{
			name: "m=10",
			exs: []TrainingExample{
				{Features: []float64{0, 1}, Target: 15},
				{Features: []float64{2, 2}, Target: 11},
				{Features: []float64{41, 1}, Target: 19},
				{Features: []float64{31, 31}, Target: 21},
				{Features: []float64{2.5, 12}, Target: 41},
				{Features: []float64{7, 12}, Target: 43},
				{Features: []float64{0, 82}, Target: 50},
				{Features: []float64{3, 17}, Target: 51},
				{Features: []float64{4, 7}, Target: 12},
				{Features: []float64{87, 2}, Target: 12},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewTrainingSet(tt.exs)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if exs := s.Examples(); !reflect.DeepEqual(exs, tt.exs) {
				t.Fatalf("want %v, got %v", tt.exs, exs)
			}
		})
	}
}

func TestNewTrainingSetInconsistentExamples(t *testing.T) {
	tests := []struct {
		name string
		exs  []TrainingExample
	}{
		{
			name: "m=10",
			exs: []TrainingExample{
				{Features: []float64{0, 1}, Target: 15},
				{Features: []float64{2, 2}, Target: 11},
				{Features: []float64{41, 1}, Target: 19},
				{Features: []float64{31, 31}, Target: 21},
				{Features: []float64{2.5, 12}, Target: 41},
				{Features: []float64{12}, Target: 43},
				{Features: []float64{0, 82}, Target: 50},
				{Features: []float64{3, 17}, Target: 51},
				{Features: []float64{4, 7}, Target: 12},
				{Features: []float64{87, 2}, Target: 12},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewTrainingSet(tt.exs); err != ErrInvalidTrainingSet {
				t.Fatalf("want %v, got %v", ErrInvalidTrainingSet, err)
			}
		})
	}
}

func TestAddDummyFeatures(t *testing.T) {
	tests := []struct {
		name string
		exs  []TrainingExample
		want []TrainingExample
	}{
		{
			name: "m=10",
			exs: []TrainingExample{
				{Features: []float64{0, 1}, Target: 15},
				{Features: []float64{2, 2}, Target: 11},
				{Features: []float64{41, 1}, Target: 19},
				{Features: []float64{31, 31}, Target: 21},
				{Features: []float64{2.5, 12}, Target: 41},
				{Features: []float64{7, 12}, Target: 43},
				{Features: []float64{0, 82}, Target: 50},
				{Features: []float64{3, 17}, Target: 51},
				{Features: []float64{4, 7}, Target: 12},
				{Features: []float64{87, 2}, Target: 12},
			},
			want: []TrainingExample{
				{Features: []float64{1, 0, 1}, Target: 15},
				{Features: []float64{1, 2, 2}, Target: 11},
				{Features: []float64{1, 41, 1}, Target: 19},
				{Features: []float64{1, 31, 31}, Target: 21},
				{Features: []float64{1, 2.5, 12}, Target: 41},
				{Features: []float64{1, 7, 12}, Target: 43},
				{Features: []float64{1, 0, 82}, Target: 50},
				{Features: []float64{1, 3, 17}, Target: 51},
				{Features: []float64{1, 4, 7}, Target: 12},
				{Features: []float64{1, 87, 2}, Target: 12},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewTrainingSet(tt.exs)
			s.AddDummyFeatures()
			// AddDummyFeatures must be idempotent.
			s.AddDummyFeatures()
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if exs := s.Examples(); !reflect.DeepEqual(s.Examples(), tt.want) {
				t.Fatalf("want %v, got %v", tt.want, exs)
			}
		})
	}
}

func TestGetDesignMatrix(t *testing.T) {
	tests := []struct {
		name string
		exs  []TrainingExample
		want [][]float64
	}{
		{
			name: "m=10",
			exs: []TrainingExample{
				{Features: []float64{0, 1}, Target: 15},
				{Features: []float64{2, 2}, Target: 11},
				{Features: []float64{41, 1}, Target: 19},
				{Features: []float64{31, 31}, Target: 21},
				{Features: []float64{2.5, 12}, Target: 41},
				{Features: []float64{7, 12}, Target: 43},
				{Features: []float64{0, 82}, Target: 50},
				{Features: []float64{3, 17}, Target: 51},
				{Features: []float64{4, 7}, Target: 12},
				{Features: []float64{87, 2}, Target: 12},
			},
			want: [][]float64{
				{0, 1},
				{2, 2},
				{41, 1},
				{31, 31},
				{2.5, 12},
				{7, 12},
				{0, 82},
				{3, 17},
				{4, 7},
				{87, 2},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewTrainingSet(tt.exs)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if x := s.GetDesignMatrix(); !reflect.DeepEqual(x, tt.want) {
				t.Fatalf("want %v, got %v", tt.want, x)
			}
		})
	}
}

func TestGetTargetVector(t *testing.T) {
	tests := []struct {
		name string
		exs  []TrainingExample
		want []float64
	}{
		{
			name: "m=10",
			exs: []TrainingExample{
				{Features: []float64{0, 1}, Target: 15},
				{Features: []float64{2, 2}, Target: 11},
				{Features: []float64{41, 1}, Target: 19},
				{Features: []float64{31, 31}, Target: 21},
				{Features: []float64{2.5, 12}, Target: 41},
				{Features: []float64{7, 12}, Target: 43},
				{Features: []float64{0, 82}, Target: 50},
				{Features: []float64{3, 17}, Target: 51},
				{Features: []float64{4, 7}, Target: 12},
				{Features: []float64{87, 2}, Target: 12},
			},
			want: []float64{15, 11, 19, 21, 41, 43, 50, 51, 12, 12},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := NewTrainingSet(tt.exs)
			if err != nil {
				t.Fatalf("want nil, got error %v", err)
			}
			if y := s.GetTargetVector(); !reflect.DeepEqual(y, tt.want) {
				t.Fatalf("want %v, got %v", tt.want, y)
			}
		})
	}
}
