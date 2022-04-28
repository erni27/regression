package regression

import (
	"reflect"
	"testing"
)

func TestTrain(t *testing.T) {
	tests := []struct {
		name     string
		fileName string
		want     Model
	}{
		{
			name:     "n=1,m=97",
			fileName: "n=1_m=97.txt",
			want:     Model{},
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
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}
