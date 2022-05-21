// Package long provides wrapper for running long operation.
package long

import "context"

// Run wraps a long running operation and handles context accordingly.
func Run[T any](ctx context.Context, f func() (T, error)) (T, error) {
	var v T
	var err error
	done := make(chan struct{})
	go func() {
		v, err = f()
		done <- struct{}{}
	}()
	select {
	case <-done:
		return v, err
	case <-ctx.Done():
		return v, ctx.Err()
	}
}
