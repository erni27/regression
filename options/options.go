// Package options defines training options for specific regression variants.
package options

// ConverganceType identifies a convergance type.
type ConverganceType int

const (
	Iterative ConverganceType = iota + 1
	Automatic
)

// GradientDescentVariant identifies a gradient descent variant.
type GradientDescentVariant int

const (
	Batch GradientDescentVariant = iota + 1
	Stochastic
)

// Options contains training options for a iterative regression algorithm.
type Options struct {
	lr  float64
	gdv GradientDescentVariant
	ct  ConverganceType
	// ci is a convergance indicator. Its interpretation depends on a convergance type.
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

// ConverganceType returns a convergance type.
func (opt Options) ConverganceType() ConverganceType {
	return opt.ct
}

// ConverganceIndicator returns a convergance indicator.
func (opt Options) ConverganceIndicator() float64 {
	return opt.ci
}

// WithIterativeConvergance returns new Options with an iterative convergance indicator.
func WithIterativeConvergance(lr float64, gdv GradientDescentVariant, i uint) Options {
	return Options{lr: lr, gdv: gdv, ct: Iterative, ci: float64(i)}
}

// WithAutomaticConvergance returns new Options with an automatic convergance indicator.
func WithAutomaticConvergance(lr float64, gdv GradientDescentVariant, t float64) Options {
	return Options{lr: lr, gdv: gdv, ct: Automatic, ci: t}
}