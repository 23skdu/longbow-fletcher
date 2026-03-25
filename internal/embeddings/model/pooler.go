package model

import (
	"github.com/23skdu/longbow-fletcher/internal/device"
)

type PoolingStrategy string

const (
	PoolingCLS  PoolingStrategy = "cls"
	PoolingMean PoolingStrategy = "mean"
	PoolingMax  PoolingStrategy = "max"
	PoolingLast PoolingStrategy = "last"
)

type Pooler struct {
	Strategy PoolingStrategy
	Backend  device.Backend
}

func NewPooler(strategy PoolingStrategy, backend device.Backend) *Pooler {
	return &Pooler{Strategy: strategy, Backend: backend}
}

func (p *Pooler) Forward(hiddenStates device.Tensor, attentionMask []int) device.Tensor {
	switch p.Strategy {
	case PoolingCLS:
		return p.poolCLS(hiddenStates)
	case PoolingMean:
		return p.poolMean(hiddenStates, attentionMask)
	case PoolingMax:
		return p.poolMax(hiddenStates, attentionMask)
	case PoolingLast:
		return p.poolLast(hiddenStates, attentionMask)
	default:
		return p.poolCLS(hiddenStates)
	}
}

func (p *Pooler) poolCLS(hiddenStates device.Tensor) device.Tensor {
	_, cols := hiddenStates.Dims()
	return hiddenStates.Slice(0, 1, 0, cols)
}

func (p *Pooler) poolMean(hiddenStates device.Tensor, attentionMask []int) device.Tensor {
	rows, cols := hiddenStates.Dims()

	data := hiddenStates.ToHost()
	result := make([]float32, cols)

	for i := 0; i < rows; i++ {
		if i < len(attentionMask) && attentionMask[i] == 0 {
			continue
		}
		for j := 0; j < cols; j++ {
			result[j] += data[i*cols+j]
		}
	}

	count := float32(rows)
	if len(attentionMask) > 0 {
		count = 0
		for _, v := range attentionMask {
			if v == 1 {
				count++
			}
		}
	}
	if count == 0 {
		count = 1
	}

	for i := range result {
		result[i] /= count
	}

	return p.Backend.NewTensor(1, cols, result)
}

func (p *Pooler) poolMax(hiddenStates device.Tensor, attentionMask []int) device.Tensor {
	rows, cols := hiddenStates.Dims()

	data := hiddenStates.ToHost()
	result := make([]float32, cols)

	for j := 0; j < cols; j++ {
		result[j] = -1e9
	}

	for i := 0; i < rows; i++ {
		if i < len(attentionMask) && attentionMask[i] == 0 {
			continue
		}
		for j := 0; j < cols; j++ {
			val := data[i*cols+j]
			if val > result[j] {
				result[j] = val
			}
		}
	}

	return p.Backend.NewTensor(1, cols, result)
}

func (p *Pooler) poolLast(hiddenStates device.Tensor, attentionMask []int) device.Tensor {
	rows, cols := hiddenStates.Dims()

	lastIdx := rows - 1
	if len(attentionMask) > 0 {
		for i := rows - 1; i >= 0; i-- {
			if i < len(attentionMask) && attentionMask[i] == 1 {
				lastIdx = i
				break
			}
		}
	}

	return hiddenStates.Slice(lastIdx, lastIdx+1, 0, cols)
}

func (p *Pooler) String() string {
	return string(p.Strategy)
}
