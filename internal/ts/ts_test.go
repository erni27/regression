package ts

import (
	"reflect"
	"testing"
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
