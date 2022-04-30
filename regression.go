package regression

// TargetType is a constraint that permits two types (float64 or integer) for target value.
// Floating point numbers are used for continous value of y, while integer corresponds to the
// discrete one.
type TargetType interface {
	~float64 | ~int
}

// TrainingExample represents a single (features, target) example.
type TrainingExample[T TargetType] struct {
	Features []float64
	Target   T
}

// TrainingSet represents set of traning examples.
type TrainingSet[T TargetType] []TrainingExample[T]

type Model[T TargetType] interface {
	Predict(x []float64) (T, error)
	Coefficients() ([]float64, error)
	Accuracy() (float64, error)
}

type Regression[T TargetType] interface {
	Run(s TrainingSet[T]) (Model[T], error)
}
