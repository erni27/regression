package regression

import (
	"encoding/csv"
	"os"
	"strconv"
)

func loadTrainingSet(name string) (TrainingSet, error) {
	f, err := os.Open("./testdata/" + name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	var ts TrainingSet = make([]TrainingExample, len(data))
	for j, line := range data {
		te := TrainingExample{Features: make([]float64, len(line)-1)}
		for i, f := range line[:len(line)-1] {
			parsed, err := strconv.ParseFloat(f, 64)
			if err != nil {
				return nil, err
			}
			te.Features[i] = parsed
		}
		parsed, err := strconv.ParseFloat(line[len(line)-1], 64)
		if err != nil {
			return nil, err
		}
		te.Target = parsed
		ts[j] = te
	}
	return ts, nil
}
