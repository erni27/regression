// Package options contains implementation of types and constants related to the regression options.
package options

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

// Options contains training options for a regression with gradient descent.
type Options struct {
	LearningRate           float64
	GradientDescentVariant GradientDescentVariant
	ConvergenceType        ConvergenceType
	ConvergenceIndicator   float64
}

// WithIterativeConvergence returns new training options with an iterative convergence indicator.
func WithIterativeConvergence(lr float64, gdv GradientDescentVariant, i uint) Options {
	return Options{LearningRate: lr, GradientDescentVariant: gdv, ConvergenceType: Iterative, ConvergenceIndicator: float64(i)}
}

// WithAutomaticConvergence returns new training options with an automatic convergence indicator.
func WithAutomaticConvergence(lr float64, gdv GradientDescentVariant, t float64) Options {
	return Options{LearningRate: lr, GradientDescentVariant: gdv, ConvergenceType: Automatic, ConvergenceIndicator: t}
}
