package regression

import "errors"

var (
	// ErrCannotConverge is returned if gradient descent cannot converge.
	// It usually means that the learning rate is too large.
	ErrCannotConverge                    = errors.New("cannot converge")
	ErrUnsupportedGradientDescentVariant = errors.New("unsupported gradient descent variant")
	ErrUnsupportedConverganceType        = errors.New("unsupported convergance type")
)

// TargetType is a constraint that permits two types (float64 or integer) for target value.
// Floating point numbers are used for continous value of y, while integer corresponds to the
// discrete one.
type TargetType interface {
	~float64 | ~int
}

// TrainingExample represents a single (features, target) example.
type TrainingExample struct {
	Features []float64
	Target   float64
}

// TrainingSet represents set of traning examples.
type TrainingSet []TrainingExample

// GetTargetVector returns the target vector from the training set.
func (s TrainingSet) GetTargetVector() []float64 {
	y := make([]float64, len(s))
	for i, te := range s {
		y[i] = te.Target
	}
	return y
}

// GetDesignMatrix returns the design matrix from the training set.
func (s TrainingSet) GetDesignMatrix() [][]float64 {
	x := make([][]float64, len(s))
	for i, te := range s {
		x[i] = te.Features
	}
	return x
}

type Model[T TargetType] interface {
	Predict(x []float64) (T, error)
	Coefficients() ([]float64, error)
	Accuracy() (float64, error)
}

type Regression[T TargetType] interface {
	Run(s TrainingSet) (Model[T], error)
}
