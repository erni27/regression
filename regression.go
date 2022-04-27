package regression

// Vector represents a n-dimensional vector of real numbers.
type Vector []float64

// TrainingExample represents a single (features, target) example.
type TrainingExample struct {
	Features Vector
	Target   float64
}

// TrainingSet represents set of traning examples.
type TrainingSet []TrainingExample
