package regression

import (
	"context"
	"encoding/csv"
	"os"
	"reflect"
	"strconv"
	"testing"
)

func TestTrainWithGradientDescentBatch(t *testing.T) {
	ctx := context.Background()
	tests := []struct {
		name         string
		dataFileName string
		opt          TrainingOptions
		want         Model
	}{
		{
			name:         "n=1, m=97, alpha=0.01, iterations=1500",
			dataFileName: "n=1_m=97.txt",
			opt:          WithConverganceAfter(NewTrainingOptions(0.01, BatchGradientDescent), 1500),
			want:         model{coefficients: []float64{-3.6303, 1.1664}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts, err := loadTrainingSet(tt.dataFileName)
			if err != nil {
				t.Fatalf("cannot load test data, got %v", err)
			}
			got, err := TrainWithGradientDescent(ctx, tt.opt, ts)
			if err != nil {
				t.Fatalf("do not want error, got %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("TrainWithGradientDescent() = %v, want %v", got, tt.want)
			}
		})
	}
}

func loadTrainingSet(name string) (TrainingSet, error) {
	f, err := os.Open("./testdata/" + name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	var ts TrainingSet = make([]TrainingExample, len(data))
	for j, line := range data {
		te := TrainingExample{Features: make([]float64, len(line)-1)}
		for i, f := range line[:len(line)-1] {
			parsed, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return nil, err
			}
			te.Features[i] = parsed
		}
		parsed, err := strconv.ParseFloat(line[len(line)-1], 64)
		if err != nil {
			return nil, err
		}
		te.Target = parsed
		ts[j] = te
	}
	return ts, nil
}
