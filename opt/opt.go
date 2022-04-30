package opt

type ConverganceType int

const (
	Iterative ConverganceType = iota + 1
	Automatic
)

type GradientDescentVariant int

const (
	Batch GradientDescentVariant = iota + 1
	Stochastic
)

type Options struct {
	lr  float64
	gdv GradientDescentVariant
	ct  ConverganceType
	// Convergance indicator value and its interpretion depend on convergance type.
	ci float64
}

func (opt Options) LearningRate() float64 {
	return opt.lr
}

func (opt Options) GradientDescentVariant() GradientDescentVariant {
	return opt.gdv
}

func (opt Options) ConverganceType() ConverganceType {
	return opt.ct
}

func (opt Options) ConverganceIndicator() float64 {
	return opt.ci
}

func WithIterativeConvergance(lr float64, gdv GradientDescentVariant, i uint) Options {
	return Options{lr: lr, gdv: gdv, ct: Iterative, ci: float64(i)}
}

func WithAutomaticConvergance(lr float64, gdv GradientDescentVariant, t float64) Options {
	return Options{lr: lr, gdv: gdv, ct: Automatic, ci: t}
}
