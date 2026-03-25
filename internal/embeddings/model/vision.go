package model

import (
	"github.com/23skdu/longbow-fletcher/internal/device"
)

type VisionConfig struct {
	ImageSize     int
	PatchSize     int
	HiddenSize    int
	NumLayers     int
	NumHeads      int
	MLPHiddenSize int
	EmbedDim      int
}

func DefaultViTBaseConfig() VisionConfig {
	return VisionConfig{
		ImageSize:     224,
		PatchSize:     16,
		HiddenSize:    768,
		NumLayers:     12,
		NumHeads:      12,
		MLPHiddenSize: 3072,
		EmbedDim:      768,
	}
}

func DefaultViTSmallConfig() VisionConfig {
	return VisionConfig{
		ImageSize:     224,
		PatchSize:     16,
		HiddenSize:    384,
		NumLayers:     12,
		NumHeads:      6,
		MLPHiddenSize: 1536,
		EmbedDim:      384,
	}
}

type VisionModel struct {
	Config        VisionConfig
	Backend       device.Backend
	PatchEmbed    device.Tensor
	PositionEmbed device.Tensor
	CLSToken      device.Tensor
	Encoder       *TransformerEncoder
	Project       device.Tensor
}

func NewVisionModel(config VisionConfig, backend device.Backend) *VisionModel {
	numPatches := (config.ImageSize / config.PatchSize) * (config.ImageSize / config.PatchSize)
	embedDim := config.HiddenSize

	m := &VisionModel{
		Config:        config,
		Backend:       backend,
		PatchEmbed:    backend.NewTensor(numPatches, embedDim, nil),
		PositionEmbed: backend.NewTensor(numPatches+1, embedDim, nil),
		CLSToken:      backend.NewTensor(1, embedDim, nil),
		Encoder:       NewTransformerEncoder(config, backend),
		Project:       backend.NewTensor(embedDim, config.EmbedDim, nil),
	}

	m.initWeights()
	return m
}

func (m *VisionModel) initWeights() {
	xavierInit(m.PatchEmbed)
	xavierInit(m.PositionEmbed)
	xavierInit(m.Project)
}

func (m *VisionModel) Forward(imagePixels []float32) device.Tensor {
	backend := m.Backend
	numPatches := (m.Config.ImageSize / m.Config.PatchSize) * (m.Config.ImageSize / m.Config.PatchSize)

	patchEmbed := backend.NewTensor(numPatches, m.Config.HiddenSize, nil)
	patchEmbed.CopyFromFloat32(imagePixels)

	positionEmbed := m.PositionEmbed.Slice(0, 1, 0, m.Config.HiddenSize)

	hiddenStates := backend.GetTensor(numPatches+1, m.Config.HiddenSize)
	hiddenStates.Paste(0, 0, patchEmbed, 0, 0, numPatches, m.Config.HiddenSize)
	hiddenStates.Paste(numPatches, 0, m.CLSToken, 0, 0, 1, m.Config.HiddenSize)
	hiddenStates.Paste(0, 0, positionEmbed, 0, 0, 1, m.Config.HiddenSize)

	encoderOutput := m.Encoder.Forward(hiddenStates)

	clsOutput := encoderOutput.Slice(numPatches, numPatches+1, 0, m.Config.HiddenSize)

	projected := backend.NewTensor(1, m.Config.EmbedDim, nil)
	projected.Mul(clsOutput, m.Project)

	backend.PutTensor(patchEmbed)
	backend.PutTensor(hiddenStates)
	backend.PutTensor(encoderOutput)
	backend.PutTensor(clsOutput)

	return projected
}

type TransformerEncoder struct {
	Config  VisionConfig
	Backend device.Backend
	Layers  []*TransformerLayer
}

type TransformerLayer struct {
	SelfAttention *MultiHeadAttention
	MLP           *VisionMLP
	LayerNorm1    *LayerNorm
	LayerNorm2    *LayerNorm
}

type VisionMLP struct {
	Config  VisionConfig
	Backend device.Backend
	FC1     device.Tensor
	FC2     device.Tensor
}

func NewTransformerEncoder(config VisionConfig, backend device.Backend) *TransformerEncoder {
	encoder := &TransformerEncoder{
		Config:  config,
		Backend: backend,
		Layers:  make([]*TransformerLayer, config.NumLayers),
	}

	for i := 0; i < config.NumLayers; i++ {
		encoder.Layers[i] = &TransformerLayer{
			SelfAttention: NewMultiHeadAttention(config, backend),
			MLP:           NewVisionMLP(config, backend),
			LayerNorm1:    NewLayerNorm(config.HiddenSize, backend),
			LayerNorm2:    NewLayerNorm(config.HiddenSize, backend),
		}
	}

	return encoder
}

func (e *TransformerEncoder) Forward(hiddenStates device.Tensor) device.Tensor {
	for _, layer := range e.Layers {
		normed1 := layer.LayerNorm1.Forward(hiddenStates)
		attnOutput := layer.SelfAttention.Forward(normed1)
		hiddenStates.Add(attnOutput)

		normed2 := layer.LayerNorm2.Forward(hiddenStates)
		mlpOutput := layer.MLP.Forward(normed2)
		hiddenStates.Add(mlpOutput)
	}
	return hiddenStates
}

func NewVisionMLP(config VisionConfig, backend device.Backend) *VisionMLP {
	mlp := &VisionMLP{
		Config:  config,
		Backend: backend,
		FC1:     backend.NewTensor(config.HiddenSize, config.MLPHiddenSize, nil),
		FC2:     backend.NewTensor(config.MLPHiddenSize, config.HiddenSize, nil),
	}
	xavierInit(mlp.FC1)
	xavierInit(mlp.FC2)
	return mlp
}

func (m *VisionMLP) Forward(x device.Tensor) device.Tensor {
	r, _ := x.Dims()
	hidden := m.Backend.NewTensor(r, m.Config.MLPHiddenSize, nil)
	hidden.Mul(x, m.FC1)
	hidden.Gelu()
	output := m.Backend.NewTensor(r, m.Config.HiddenSize, nil)
	output.Mul(hidden, m.FC2)
	return output
}

type MultiHeadAttention struct {
	Config  VisionConfig
	Backend device.Backend
	Query   device.Tensor
	Key     device.Tensor
	Value   device.Tensor
	OutProj device.Tensor
}

func NewMultiHeadAttention(config VisionConfig, backend device.Backend) *MultiHeadAttention {
	attn := &MultiHeadAttention{
		Config:  config,
		Backend: backend,
		Query:   backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Key:     backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Value:   backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		OutProj: backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
	}
	xavierInit(attn.Query)
	xavierInit(attn.Key)
	xavierInit(attn.Value)
	xavierInit(attn.OutProj)
	return attn
}

func (a *MultiHeadAttention) Forward(x device.Tensor) device.Tensor {
	r, c := x.Dims()
	q := a.Backend.NewTensor(r, c, nil)
	k := a.Backend.NewTensor(r, c, nil)
	v := a.Backend.NewTensor(r, c, nil)

	q.Mul(x, a.Query)
	k.Mul(x, a.Key)
	v.Mul(x, a.Value)

	headDim := c / a.Config.NumHeads

	attnScores := a.Backend.GetTensor(r, r)
	kT := k.T()
	attnScores.Mul(q, kT)
	attnScores.Scale(1.0 / float32(headDim))
	attnScores.Softmax()

	attnOutput := a.Backend.NewTensor(r, c, nil)
	attnOutput.Mul(attnScores, v)

	projected := a.Backend.NewTensor(r, c, nil)
	projected.Mul(attnOutput, a.OutProj)

	return projected
}
