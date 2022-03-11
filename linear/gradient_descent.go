package linear

import (
	"context"
	"math"

	"github.com/erni27/regression"
)

// TrainingExample represents a single (features, target) example.
type TrainingExample struct {
	Features regression.Vector
	Target   float64
}

// TrainingSet represents set of traning examples.
//
// This is an alias for slice of TraningExample struct.
type TrainingSet []TrainingExample

func (m *Model) Train(ts TrainingSet, iterations int) error {
	n := len(ts[0].Features)
	var curr, next regression.Vector = make([]float64, n+1), make([]float64, n+1)
	for i := 0; i < iterations; i++ {
		for j := 0; j < n+1; j++ {
			g, err := calcGradient(ts, curr, j)
			if err != nil {
				return err
			}
			step := m.learningRate * g
			next[j] = curr[j] - step
			if math.IsNaN(next[j]) || math.IsInf(next[j], 0) {
				return ErrCannotConverge
			}
		}
		copy(curr, next)
	}
	m.coefficients = curr
	return nil
}

func (m *Model) TrainWithAutomaticConvergence(ctx context.Context, ts TrainingSet, t float64) error {
	type result struct {
		coeff regression.Vector
		err   error
	}
	ch := make(chan result)
	n := len(ts[0].Features)
	var curr, next regression.Vector = make([]float64, n+1), make([]float64, n+1)
	go func() {
		<-ctx.Done()
		ch <- result{coeff: curr, err: ctx.Err()}
	}()
	go func() {
		for {
			for j := 0; j < n+1; j++ {
				g, err := calcGradient(ts, curr, j)
				if err != nil {
					ch <- result{coeff: curr, err: ErrCannotConverge}
					return
				}
				step := m.learningRate * g
				next[j] = curr[j] - step
			}
			c, err := hasBeenConverged(ts, curr, next, t)
			if err != nil {
				ch <- result{coeff: curr, err: ErrCannotConverge}
				return
			}
			if c {
				ch <- result{coeff: curr, err: nil}
				return
			}
			copy(curr, next)
		}
	}()
	r := <-ch
	if r.err != nil {
		return r.err
	}
	m.coefficients = r.coeff
	return nil
}

func calcGradient(ts TrainingSet, coeff regression.Vector, j int) (float64, error) {
	var g float64
	var calc, base func(te TrainingExample) (float64, error)
	base = func(te TrainingExample) (float64, error) {
		y, err := calcHypho(te.Features, coeff)
		if err != nil {
			return 0, err
		}
		return y - te.Target, nil
	}
	if j != 0 {
		calc = func(te TrainingExample) (float64, error) {
			baseY, err := base(te)
			if err != nil {
				return 0, nil
			}
			return baseY * te.Features[j-1], nil
		}
	} else {
		calc = base
	}
	for _, v := range ts {
		r, err := calc(v)
		if err != nil {
			return 0, nil
		}
		g += r
	}
	return g / float64(len(ts)), nil
}

func hasBeenConverged(ts TrainingSet, oldCoeffs, newCoeffs regression.Vector, t float64) (bool, error) {
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

func calcCost(ts TrainingSet, coeff regression.Vector) (float64, error) {
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
