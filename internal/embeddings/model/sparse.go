package model

import (
	"github.com/23skdu/longbow-fletcher/internal/device"
)

type SparseConfig struct {
	BertConfig
	VocabSize int
}

func DefaultSparseConfig() SparseConfig {
	return SparseConfig{
		BertConfig: DefaultBertTinyConfig(),
		VocabSize:  30522,
	}
}

type SparseModel struct {
	Config    SparseConfig
	Backend   device.Backend
	BertModel *BertModel
	MLPHead   device.Tensor
}

func NewSparseModel(config SparseConfig, backend device.Backend) *SparseModel {
	return &SparseModel{
		Config:    config,
		Backend:   backend,
		BertModel: NewBertModelWithBackend(config.BertConfig, backend),
		MLPHead:   backend.NewTensor(config.HiddenSize, config.VocabSize, nil),
	}
}

type SparseEmbedding struct {
	Weights map[int]float32
	Dim     int
}

func NewSparseEmbedding(vocabSize int) *SparseEmbedding {
	return &SparseEmbedding{
		Weights: make(map[int]float32),
		Dim:     vocabSize,
	}
}

func (s *SparseEmbedding) Set(tokenID int, weight float32) {
	if weight > 0 {
		s.Weights[tokenID] = weight
	}
}

func (s *SparseEmbedding) Get(tokenID int) float32 {
	return s.Weights[tokenID]
}

func (s *SparseEmbedding) ToDense() []float32 {
	dense := make([]float32, s.Dim)
	for id, w := range s.Weights {
		dense[id] = w
	}
	return dense
}

func (s *SparseEmbedding) NonZeroCount() int {
	count := 0
	for _, w := range s.Weights {
		if w > 0 {
			count++
		}
	}
	return count
}

type SpladeEncoder struct {
	Config    SparseConfig
	Backend   device.Backend
	BertModel *BertModel
	MLPHead   device.Tensor
}

func NewSpladeEncoder(config SparseConfig, backend device.Backend) *SpladeEncoder {
	return &SpladeEncoder{
		Config:    config,
		Backend:   backend,
		BertModel: NewBertModelWithBackend(config.BertConfig, backend),
		MLPHead:   backend.NewTensor(config.HiddenSize, config.VocabSize, nil),
	}
}

func (e *SpladeEncoder) Forward(inputIDs []int, lengths []int) *SparseEmbedding {
	hiddenStates := e.BertModel.ForwardBatch(inputIDs, lengths)

	rows, _ := hiddenStates.Dims()
	vocabSize := e.Config.VocabSize

	mlpOut := e.Backend.NewTensor(rows, vocabSize, nil)
	mlpOut.Mul(hiddenStates, e.MLPHead)

	emb := NewSparseEmbedding(vocabSize)

	data := mlpOut.ToHost()
	for i := 0; i < rows; i++ {
		for j := 0; j < vocabSize && j < len(data)-i*vocabSize; j++ {
			idx := i*vocabSize + j
			if idx < len(data) {
				val := data[idx]
				if val > 0 {
					val = val * 0.01
					if val > 0 {
						existing := emb.Get(j)
						if val > existing {
							emb.Set(j, val)
						}
					}
				}
			}
		}
	}

	e.Backend.PutTensor(mlpOut)

	return emb
}
