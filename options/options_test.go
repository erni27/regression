package options

import (
	"testing"
)

func TestWithAutomaticConvergence(t *testing.T) {
	tests := []struct {
		name string
		lr   float64
		gdv  GradientDescentVariant
		t    float64
	}{
		{name: "batch_alpha=0.001_threshold=0.001", lr: 0.001, gdv: Batch, t: 0.001},
		{name: "stochastic_alpha=0.03_threshold=0.01", lr: 0.03, gdv: Stochastic, t: 0.01},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := WithAutomaticConvergence(tt.lr, tt.gdv, tt.t)
			if lr := o.LearningRate(); lr != tt.lr {
				t.Errorf("want %f, got %f", tt.lr, lr)
			}
			if gdv := o.GradientDescentVariant(); gdv != tt.gdv {
				t.Errorf("want %d, got %d", tt.gdv, gdv)
			}
			if ct := o.ConvergenceType(); ct != Automatic {
				t.Errorf("want %d, got %d", Automatic, ct)
			}
			if ci := o.ConvergenceIndicator(); ci != tt.t {
				t.Errorf("want %f, got %f", tt.t, ci)
			}
		})
	}
}

func TestWithIterativeConvergence(t *testing.T) {
	tests := []struct {
		name string
		lr   float64
		gdv  GradientDescentVariant
		i    uint
	}{
		{name: "batch_alpha=0.001_iterations=1000", lr: 0.001, gdv: Batch, i: 1000},
		{name: "stochastic_alpha=0.03_iterations=100000", lr: 0.03, gdv: Stochastic, i: 100000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := WithIterativeConvergence(tt.lr, tt.gdv, tt.i)
			if lr := o.LearningRate(); lr != tt.lr {
				t.Errorf("want %f, got %f", tt.lr, lr)
			}
			if gdv := o.GradientDescentVariant(); gdv != tt.gdv {
				t.Errorf("want %d, got %d", tt.gdv, gdv)
			}
			if ct := o.ConvergenceType(); ct != Iterative {
				t.Errorf("want %d, got %d", Iterative, ct)
			}
			if ci := o.ConvergenceIndicator(); uint(ci) != tt.i {
				t.Errorf("want %d, got %f", tt.i, ci)
			}
		})
	}
}
