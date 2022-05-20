package options

import (
	"testing"
)

func TestWithAutomaticConvergance(t *testing.T) {
	tests := []struct {
		name string
		lr   float64
		gdv  GradientDescentVariant
		fst  FeatureScalingTechnique
		t    float64
	}{
		{name: "batch_alpha=0.001_threshold=0.001", lr: 0.001, gdv: Batch, fst: Standarization, t: 0.001},
		{name: "stochastic_alpha=0.03_threshold=0.01", lr: 0.03, gdv: Stochastic, fst: Normalization, t: 0.01},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := WithAutomaticConvergance(tt.lr, tt.gdv, tt.fst, tt.t)
			if lr := o.LearningRate(); lr != tt.lr {
				t.Errorf("want %f, got %f", tt.lr, lr)
			}
			if gdv := o.GradientDescentVariant(); gdv != tt.gdv {
				t.Errorf("want %d, got %d", tt.gdv, gdv)
			}
			if fst := o.FeatureScalingTechnique(); fst != tt.fst {
				t.Errorf("want %d, got %d", tt.fst, fst)
			}
			if ct := o.ConverganceType(); ct != Automatic {
				t.Errorf("want %d, got %d", Automatic, ct)
			}
			if ci := o.ConverganceIndicator(); ci != tt.t {
				t.Errorf("want %f, got %f", tt.t, ci)
			}
		})
	}
}

func TestWithIterativeConvergance(t *testing.T) {
	tests := []struct {
		name string
		lr   float64
		gdv  GradientDescentVariant
		fst  FeatureScalingTechnique
		i    uint
	}{
		{name: "batch_alpha=0.001_iterations=1000", lr: 0.001, gdv: Batch, fst: None, i: 1000},
		{name: "stochastic_alpha=0.03_iterations=100000", lr: 0.03, gdv: Stochastic, fst: Normalization, i: 100000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := WithIterativeConvergance(tt.lr, tt.gdv, tt.fst, tt.i)
			if lr := o.LearningRate(); lr != tt.lr {
				t.Errorf("want %f, got %f", tt.lr, lr)
			}
			if gdv := o.GradientDescentVariant(); gdv != tt.gdv {
				t.Errorf("want %d, got %d", tt.gdv, gdv)
			}
			if fst := o.FeatureScalingTechnique(); fst != tt.fst {
				t.Errorf("want %d, got %d", tt.fst, fst)
			}
			if ct := o.ConverganceType(); ct != Iterative {
				t.Errorf("want %d, got %d", Iterative, ct)
			}
			if ci := o.ConverganceIndicator(); uint(ci) != tt.i {
				t.Errorf("want %d, got %f", tt.i, ci)
			}
		})
	}
}
