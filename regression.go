package regression

import "errors"

var (
	// ErrCannotConverge is returned if gradient descent cannot converge.
	// It usually means that the learning rate is too large.
	ErrCannotConverge                    = errors.New("cannot converge")
	ErrUnsupportedGradientDescentVariant = errors.New("unsupported gradient descent variant")
	ErrUnsupportedConverganceType        = errors.New("unsupported convergance type")
	ErrInvalidTrainingSet                = errors.New("invalid training set")
	ErrInvalidFeatureVector              = errors.New("invalid feature vector")
)

// TargetType is a constraint that permits two types (float64 or integer) for target value.
// Floating point numbers are used for continous value of y, while integer corresponds to the
// discrete one.
type TargetType interface {
	~float64 | ~int
}

type Model[T TargetType] interface {
	Predict(x []float64) (T, error)
	Coefficients() []float64
	Accuracy() float64
}

type Regression[T TargetType] interface {
	Run(s TrainingSet) (Model[T], error)
}

// TrainingExample represents a single (features, target) example.
type TrainingExample struct {
	Features []float64
	Target   float64
}

// NewTrainingSet creates a new training set if examples are valid in a current context.
// It doesn't check for correctness of values. If examples are invalid, returns an error.
func NewTrainingSet(exs []TrainingExample) (*TrainingSet, error) {
	if !areExamplesValid(exs) {
		return nil, ErrInvalidTrainingSet
	}
	return &TrainingSet{examples: exs}, nil
}

// TrainingSet represents set of traning examples.
type TrainingSet struct {
	examples []TrainingExample
	x        [][]float64
	y        []float64
	// isDummyAdded is a flag indicating if a dummy feature was added to each training example.
	isDummyAdded bool
}

// Examples returns all training examples from training set.
func (s *TrainingSet) Examples() []TrainingExample {
	return s.examples
}

// GetTargetVector returns the target vector from the training set.
func (s *TrainingSet) GetTargetVector() []float64 {
	if s.y != nil {
		return s.y
	}
	y := make([]float64, len(s.examples))
	for i, te := range s.examples {
		y[i] = te.Target
	}
	s.y = y
	return y
}

// GetDesignMatrix returns the design matrix from the training set.
func (s *TrainingSet) GetDesignMatrix() [][]float64 {
	if s.x != nil {
		return s.x
	}
	x := make([][]float64, len(s.examples))
	for i, te := range s.examples {
		x[i] = te.Features
	}
	s.x = x
	return x
}

// AddDummyFeatures puts dummy feature (equals 1) for each training example being a part of the training set.
func (s *TrainingSet) AddDummyFeatures() {
	if s.isDummyAdded {
		return
	}
	for i := 0; i < len(s.examples); i++ {
		s.examples[i].Features = append([]float64{1}, s.examples[i].Features...)
	}
	s.isDummyAdded = true
}

func areExamplesValid(exs []TrainingExample) bool {
	if exs == nil || len(exs) == 0 {
		return false
	}
	// n represents number of features.
	// This number must be consistent across all features vectors.
	n := len(exs[0].Features)
	for i := 1; i < len(exs); i++ {
		if len(exs[i].Features) != n {
			return false
		}
	}
	return true
}
