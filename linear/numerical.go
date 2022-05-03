package linear

import (
	"github.com/erni27/regression"
	"github.com/erni27/regression/internal/gd"
	"github.com/erni27/regression/opt"
)

// WithGradientDescent initializes linear regression with numerical approach.
// It finds the value of coefficients by taking steps in each iteration towards
// the minimum of a cost function (LMS).
func WithGradientDescent(o opt.Options) regression.Regression[float64] {
	var f regressionFunc = func(s regression.TrainingSet) (regression.Model[float64], error) {
		return numerical(o, s)
	}
	return f
}

// numerical runs linear regression for given training set. It uses an numerical approach
// for computing coefficients (gradient descent).
func numerical(o opt.Options, s regression.TrainingSet) (regression.Model[float64], error) {
	s.AddDummyFeatures()
	x := s.GetDesignMatrix()
	y := s.GetTargetVector()

	// Init stepper.
	var gds gd.Stepper
	switch o.GradientDescentVariant() {
	case opt.Batch:
		gds = gd.NewBatchStepper(hyphothesis, x, y, o.LearningRate())
	case opt.Stochastic:
		gds = gd.NewStochasticStepper(hyphothesis, x, y, o.LearningRate())
	default:
		return nil, regression.ErrUnsupportedGradientDescentVariant
	}

	var coeffs []float64
	var err error
	switch o.ConverganceType() {
	case opt.Iterative:
		coeffs, err = gd.ConvergeAfter(gds, int(o.ConverganceIndicator()))
	case opt.Automatic:
		coeffs, err = gd.ConvergeAutomatically(gds, cost, o.ConverganceIndicator())
	default:
		return nil, regression.ErrUnsupportedConverganceType
	}
	if err != nil {
		return nil, err
	}
	r2, err := calcR2(s, coeffs)
	if err != nil {
		return nil, err
	}
	return model{coeffs: coeffs, r2: r2}, nil
}
