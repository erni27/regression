package regression

import (
	"context"
	"errors"
	"fmt"
	"math"
)

type optimisationMethod int

// Discriminators that allow distinguish between optimisation types.
const (
	BatchGradientDescent optimisationMethod = iota + 1
	StochasticGradientDescent
)

type TrainingOptions struct {
	learningRate       float64
	optimisationMethod optimisationMethod
	converganceOpt     converganceOptions
}

func NewTrainingOptions(learningRate float64, optimisationMethod optimisationMethod) TrainingOptions {
	return TrainingOptions{learningRate: learningRate, optimisationMethod: optimisationMethod}
}

func WithConverganceAfter(opt TrainingOptions, n uint) TrainingOptions {
	c := converganceOptions{converganceType: iterations, value: float64(n)}
	opt.converganceOpt = c
	return opt
}

func WithAutomaticConvergance(opt TrainingOptions, t float64) TrainingOptions {
	c := converganceOptions{converganceType: automatic, value: t}
	opt.converganceOpt = c
	return opt
}

func (opt TrainingOptions) LearningRate() float64 {
	return opt.learningRate
}

func (opt TrainingOptions) OptimisationMethod() optimisationMethod {
	return opt.optimisationMethod
}

func (opt TrainingOptions) String() string {
	var o, c string
	switch opt.optimisationMethod {
	case BatchGradientDescent:
		o = "batch gradient descent"
	case StochasticGradientDescent:
		o = "stochastic gradient descent"
	default:
		o = "not supported optimisation method"
	}
	switch opt.converganceOpt.converganceType {
	case automatic:
		c = fmt.Sprintf("automatic convergance, threshould=%f", opt.converganceOpt.value)
	case iterations:
		c = fmt.Sprintf("convergance after %0.f", opt.converganceOpt.value)
	default:
		c = "convergance options not set"
	}
	return fmt.Sprintf("learning rate: %v, optimisation: %v, convergance: %v", opt.learningRate, o, c)
}

type converganceOptions struct {
	converganceType converganceType
	value           float64
}

type converganceType int

const (
	automatic converganceType = iota + 1
	iterations
)

var (
	// ErrCannotConverge indicates an issue with convergance if learning rate is too big.
	ErrCannotConverge = errors.New("cannot converge")
	// ErrUnsupportedVariant indicates the usage of unsupported gradient descent algorithm's variant.
	ErrUnsupportedVariant = errors.New("unsupported gradient descent variant")
)

// TrainWithGradientDescent runs linear regression. It uses gradient descent optimisation algorithm.
// The options indicate gradient descent variant and convergance options.
// Returns a trained model if training succeeded or a non-nil error if not.
//
//
func TrainWithGradientDescent(ctx context.Context, opt TrainingOptions, ts TrainingSet) (Model, error) {
	type result struct {
		coeffs Vector
		err    error
	}
	c := make(chan result, 1)
	escape := false
	go func() {
		iterations := int(opt.converganceOpt.value)
		var coeffs Vector = make([]float64, len(ts[0].Features)+1)
		var err error
		for i := 0; i < iterations; i++ {
			coeffs, err = calcCoefficients(coeffs, ts, opt.learningRate)
			if err != nil {
				c <- result{coeffs, err}
				return
			}
			if escape {
				return
			}
		}
		c <- result{coeffs, nil}
	}()
	select {
	case r := <-c:
		if r.err != nil {
			return nil, r.err
		}
		return model{coeffs: r.coeffs}, nil
	case <-ctx.Done():
		escape = true
		return nil, ctx.Err()
	}
}

func calcCoefficients(coeffs Vector, ts TrainingSet, lr float64) (Vector, error) {
	var r Vector = make([]float64, len(coeffs))
	n := len(ts[0].Features)
	for j := 0; j < n+1; j++ {
		g, err := calcPartialDerivative(ts, coeffs, j)
		if err != nil {
			return r, err
		}
		r[j] = coeffs[j] - lr*g
		if math.IsNaN(r[j]) || math.IsInf(r[j], 0) {
			return r, ErrCannotConverge
		}
	}
	return r, nil
}

// func (t RegressionWithGradientDescent) TrainWithAutomaticConvergence(
// 	ctx context.Context,
// 	ts TrainingSet,
// 	threshold float64,
// ) (Model, error) {
// 	type result struct {
// 		coeff Vector
// 		err   error
// 	}
// 	ch := make(chan result)
// 	n := len(ts[0].Features)
// 	var curr, next Vector = make([]float64, n+1), make([]float64, n+1)
// 	go func() {
// 		<-ctx.Done()
// 		ch <- result{coeff: curr, err: ctx.Err()}
// 	}()
// 	go func() {
// 		for {
// 			for j := 0; j < n+1; j++ {
// 				g, err := calcBatchGradientStep(ts, curr, j)
// 				if err != nil {
// 					ch <- result{coeff: curr, err: ErrCannotConverge}
// 					return
// 				}
// 				step := t.learningRate * g
// 				next[j] = curr[j] - step
// 			}
// 			c, err := hasBeenConverged(ts, curr, next, threshold)
// 			if err != nil {
// 				ch <- result{coeff: curr, err: ErrCannotConverge}
// 				return
// 			}
// 			if c {
// 				ch <- result{coeff: curr, err: nil}
// 				return
// 			}
// 			copy(curr, next)
// 		}
// 	}()
// 	r := <-ch
// 	if r.err != nil {
// 		return nil, r.err
// 	}
// 	return model{coefficients: r.coeff}, nil
// }

// calcPartialDerivative calculates partial derivative of a cost function (LMS).
func calcPartialDerivative(ts TrainingSet, coeff Vector, j int) (float64, error) {
	// Derivative calculation differs if j equals zero.
	// In that case dummy feature should be taken into an account.
	calculate := func(te TrainingExample) (float64, error) {
		y, err := calcHypho(te.Features, coeff)
		if err != nil {
			return 0, err
		}
		// Local derivative equals hyphothesis for given training example
		// minus the target value multiplied by the dummy feature which equals 1,
		// so is ommitted.
		return y - te.Target, nil
	}
	if j > 0 {
		base := calculate
		calculate = func(te TrainingExample) (float64, error) {
			r, err := base(te)
			if err != nil {
				return 0, nil
			}
			// For j > 0 local derivative equals hyphothesis for given training example
			// minus the target value multiplied by the j-1 feature.
			return r * te.Features[j-1], nil
		}
	}

	var g float64
	for _, v := range ts {
		r, err := calculate(v)
		if err != nil {
			return 0, nil
		}
		g += r
	}
	return g / float64(len(ts)), nil
}

// hasBeenConverged performs convergence test.
func hasBeenConverged(ts TrainingSet, oldCoeffs, newCoeffs Vector, t float64) (bool, error) {
	oldCost, err := calcCost(ts, oldCoeffs)
	if err != nil {
		return false, err
	}
	newCost, err := calcCost(ts, newCoeffs)
	if err != nil {
		return false, err
	}
	ratio := newCost / oldCost
	if ratio > 1 {
		return false, ErrCannotConverge
	}
	return 1-ratio < t, nil
}

// calcCost calculates cost function value for given training set and coefficients.
func calcCost(ts TrainingSet, coeff Vector) (float64, error) {
	var c float64
	for _, v := range ts {
		h, err := calcHypho(v.Features, coeff)
		if err != nil {
			return 0, err
		}
		c += math.Pow(h-v.Target, 2)
	}
	return c / float64((2 * len(ts))), nil
}
