// Package options contains implementation of types and constants related to the regression options.
package options

// FeatureScalingTechnique identifies a feature scaling technique.
type FeatureScalingTechnique int

const (
	None FeatureScalingTechnique = iota
	Normalization
	Standarization
)

// ConvergenceType identifies a convergence type.
type ConvergenceType int

const (
	Iterative ConvergenceType = iota + 1
	Automatic
)

// GradientDescentVariant identifies a gradient descent variant.
type GradientDescentVariant int

const (
	Batch GradientDescentVariant = iota + 1
	Stochastic
)

// Options contains training options for an iterative regression algorithm.
type Options struct {
	lr  float64
	gdv GradientDescentVariant
	ct  ConvergenceType
	// ci is a convergence indicator. Its interpretation depends on a convergence type.
	ci float64
}

// LearningRate returns a learning rate.
func (opt Options) LearningRate() float64 {
	return opt.lr
}

// GradientDescentVariant returns a gradient descent variant.
func (opt Options) GradientDescentVariant() GradientDescentVariant {
	return opt.gdv
}

// ConvergenceType returns a convergence type.
func (opt Options) ConvergenceType() ConvergenceType {
	return opt.ct
}

// ConvergenceIndicator returns a convergence indicator.
func (opt Options) ConvergenceIndicator() float64 {
	return opt.ci
}

// WithIterativeConvergence returns new Options with an iterative convergence indicator.
func WithIterativeConvergence(lr float64, gdv GradientDescentVariant, i uint) Options {
	return Options{lr: lr, gdv: gdv, ct: Iterative, ci: float64(i)}
}

// WithAutomaticConvergence returns new Options with an automatic convergence indicator.
func WithAutomaticConvergence(lr float64, gdv GradientDescentVariant, t float64) Options {
	return Options{lr: lr, gdv: gdv, ct: Automatic, ci: t}
}
