package regression

import (
	"encoding/csv"
	"os"
	"strconv"

	"golang.org/x/exp/constraints"
)

func loadTrainingSet[float constraints.Float](name string) (TrainingSet[float], error) {
	f, err := os.Open("./testdata/" + name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	var ts TrainingSet[float] = make([]TrainingExample[float], len(data))
	for j, line := range data {
		te := TrainingExample[float]{Features: make([]float, len(line)-1)}
		for i, f := range line[:len(line)-1] {
			parsed, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return nil, err
			}
			te.Features[i] = float(parsed)
		}
		parsed, err := strconv.ParseFloat(line[len(line)-1], 64)
		if err != nil {
			return nil, err
		}
		te.Target = float(parsed)
		ts[j] = te
	}
	return ts, nil
}
