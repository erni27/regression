//go:build !race
// +build !race

package long

import (
	"context"
	"testing"
)

func TestRun(t *testing.T) {
	want := 997
	ctx := context.Background()
	got, err := Run(ctx, func() (int, error) { return want, nil })
	if err != nil {
		t.Fatalf("want nil, got error %v", err)
	}
	if got != want {
		t.Fatalf("want %d, got error %d", want, got)
	}
}

// TestRun_ContextCanceled fails under the data race detector due to immediate return after the ctx.Done().
func TestRun_ContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Run(ctx, func() (int, error) { return 997, nil })
	if err != context.Canceled {
		t.Fatalf("want %v, got error %v", context.Canceled, err)
	}
}
