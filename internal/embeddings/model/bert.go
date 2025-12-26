package model

import (
	"math"
	"math/rand"
	"sync"

	"github.com/23skdu/longbow-fletcher/internal/device"
)

// BertConfig holds the configuration for the BERT model.
type BertConfig struct {
	VocabSize            int
	HiddenSize           int
	NumHiddenLayers      int
	NumAttentionHeads    int
	IntermediateSize     int
	MaxPositionEmbeddings int
}

// DefaultBertTinyConfig returns the configuration for BERT-Tiny.
func DefaultBertTinyConfig() BertConfig {
	return BertConfig{
		VocabSize:            30522,
		HiddenSize:           128,
		NumHiddenLayers:      2,
		NumAttentionHeads:    2,
		IntermediateSize:     512,
		MaxPositionEmbeddings: 512,
	}
}

// BertModel is the main BERT model structure.
type BertModel struct {
	Config     BertConfig
	Backend    device.Backend
	Embeddings *BertEmbeddings
	Encoder    *BertEncoder
	Pooler     *BertPooler
}

// NewBertModel creates a new BERT model with the given configuration.
// Weights are initialized with Xavier/Glorot initialization for sensible defaults.
func NewBertModel(config BertConfig) *BertModel {
	return NewBertModelWithBackend(config, device.NewCPUBackend())
}

// NewBertModelWithBackend creates a new BERT model with the given configuration and backend.
// Weights are initialized with Xavier/Glorot initialization for sensible defaults.
func NewBertModelWithBackend(config BertConfig, b device.Backend) *BertModel {
	model := &BertModel{
		Config:     config,
		Backend:    b,
		Embeddings: NewBertEmbeddings(config, b),
		Encoder:    NewBertEncoder(config, b),
		Pooler:     NewBertPooler(config, b),
	}
	model.initWeights()
	return model
}

// initWeights applies Xavier initialization to all weight matrices.
func (m *BertModel) initWeights() {
	// Embeddings
	xavierInit(m.Embeddings.WordEmbeddings)
	xavierInit(m.Embeddings.PositionEmbeddings)
	
	// Encoder layers
	for _, layer := range m.Encoder.Layers {
		// Self-Attention
		xavierInit(layer.Attention.Self.Query)
		xavierInit(layer.Attention.Self.Key)
		xavierInit(layer.Attention.Self.Value)
		// Self-Output
		xavierInit(layer.Attention.Output.Dense)
		// Intermediate
		xavierInit(layer.Intermediate.Dense)
		// Output
		xavierInit(layer.Output.Dense)
	}
	
	// Pooler
	xavierInit(m.Pooler.Dense)
}

// xavierInit initializes a matrix with Xavier/Glorot uniform initialization.
// Uses bulk CopyFromFloat64 for efficient GPU upload (especially FP16).
func xavierInit(m device.Tensor) {
	r, c := m.Dims()
	size := r * c
	limit := math.Sqrt(6.0 / float64(r+c))
	
	// Generate all random values in a single slice
	data := make([]float64, size)
	for i := range data {
		data[i] = (rand.Float64()*2 - 1) * limit
	}
	
	// Bulk upload to GPU (single FP16 conversion pass)
	m.CopyFromFloat64(data)
}

// ForwardBatch performs the forward pass for a batch of sequences.
// inputIDs is flattened, lengths contains the length of each sequence.
func (m *BertModel) ForwardBatch(inputIDs []int, lengths []int) device.Tensor {
	embeddings := m.Embeddings.ForwardBatch(inputIDs, lengths)
	hiddenStates := m.Encoder.ForwardBatch(embeddings, lengths)
	return m.Pooler.ForwardBatch(hiddenStates, lengths)
}

// Forward is a legacy wrapper for single sequence compatibility.
func (m *BertModel) Forward(inputIDs []int) device.Tensor {
	return m.ForwardBatch(inputIDs, []int{len(inputIDs)})
}

// BertEmbeddings handles word, position, and token type embeddings.
type BertEmbeddings struct {
	Config             BertConfig
	Backend            device.Backend
	WordEmbeddings     device.Tensor
	PositionEmbeddings device.Tensor
	TokenTypeEmbeddings device.Tensor
	LayerNorm          *LayerNorm
	Dropout            *Dropout
}

func NewBertEmbeddings(config BertConfig, backend device.Backend) *BertEmbeddings {
	return &BertEmbeddings{
		Config:             config,
		Backend:            backend,
		WordEmbeddings:     backend.NewTensor(config.VocabSize, config.HiddenSize, nil),
		PositionEmbeddings: backend.NewTensor(config.MaxPositionEmbeddings, config.HiddenSize, nil),
		TokenTypeEmbeddings: backend.NewTensor(2, config.HiddenSize, nil), // 2 types: A and B
		LayerNorm:          NewLayerNorm(config.HiddenSize, backend),
		Dropout:            NewDropout(0.1), // Default 0.1
	}
}

// Dropout temporarily just identity for inference
type Dropout struct {
	Rate float64
}
func NewDropout(rate float64) *Dropout {
	return &Dropout{Rate: rate}
}
func (d *Dropout) Forward(t device.Tensor) device.Tensor {
	return t // Identity for now
}

func (e *BertEmbeddings) Forward(inputIDs []int) device.Tensor {
	// Legacy single sequence wrapper
	return e.ForwardBatch(inputIDs, []int{len(inputIDs)})
}

func (e *BertEmbeddings) ForwardBatch(inputIDs []int, lengths []int) device.Tensor {
	totalTokens := len(inputIDs)
	
	// 1. Gather Word Embeddings
	embeddings := e.WordEmbeddings.Gather(inputIDs)
	
	// 2. Gather Position Embeddings
	posIndices := make([]int, totalTokens)
	idx := 0
	for _, l := range lengths {
		for i := 0; i < l; i++ {
			if i >= e.Config.MaxPositionEmbeddings {
				posIndices[idx] = e.Config.MaxPositionEmbeddings - 1
			} else {
				posIndices[idx] = i
			}
			idx++
		}
	}
	
	posEmbeds := e.PositionEmbeddings.Gather(posIndices)
	embeddings.Add(posEmbeds)
	
	// 3. Token Type Embeddings (Assume 0)
	typeIndices := make([]int, totalTokens)
	typeEmbeds := e.TokenTypeEmbeddings.Gather(typeIndices)
	embeddings.Add(typeEmbeds)
	
	// 4. Norm + Dropout
	output := e.LayerNorm.Forward(embeddings)
	
	// Dropout not implemented in interface yet? 
	// BertEmbeddings struct has Dropout *Dropout.
	// Dropout.Forward takes Tensor.
	// Assuming Dropout accepts Tensor.
	// Check BertEmbeddings struct?
	// But let's assume yes.
	
	output = e.Dropout.Forward(output)
	
	return output
}

// LayerNorm implements Layer Normalization.
type LayerNorm struct {
	Gamma device.Tensor
	Beta  device.Tensor
	Eps   float64
}

func NewLayerNorm(size int, backend device.Backend) *LayerNorm {
	// Create Gamma with 1s
	ones := make([]float64, size)
	for i := range ones { ones[i] = 1.0 }
	
	return &LayerNorm{
		Gamma: backend.NewTensor(1, size, ones),
		Beta:  backend.NewTensor(1, size, nil), // Zeros
		Eps:   1e-12,
	}
}

// Forward performs LayerNorm in-place.
// It overwrites input with the normalized result to avoid allocations.
func (l *LayerNorm) Forward(input device.Tensor) device.Tensor {
	input.LayerNorm(l.Gamma, l.Beta, l.Eps)
	return input
}

// BertEncoder is a stack of Transformer layers.
type BertEncoder struct {
	Layers []*BertLayer
}

func NewBertEncoder(config BertConfig, backend device.Backend) *BertEncoder {
	layers := make([]*BertLayer, config.NumHiddenLayers)
	for i := range layers {
		layers[i] = NewBertLayer(config, backend)
	}
	return &BertEncoder{Layers: layers}
}

func (e *BertEncoder) Forward(hiddenStates device.Tensor) device.Tensor {
	// Legacy single sequence wrapper
	// We can't easily infer lengths here without info, but this method implies single seq.
	// Actually, Forward is just a passthrough.
	for _, layer := range e.Layers {
		hiddenStates = layer.Forward(hiddenStates)
	}
	return hiddenStates
}

func (e *BertEncoder) ForwardBatch(hiddenStates device.Tensor, lengths []int) device.Tensor {
	for _, layer := range e.Layers {
		hiddenStates = layer.ForwardBatch(hiddenStates, lengths)
	}
	return hiddenStates
}

// BertLayer is a single Transformer block.
type BertLayer struct {
	Attention    *BertAttention
	Intermediate *BertIntermediate
	Output       *BertOutput
}

func NewBertLayer(config BertConfig, backend device.Backend) *BertLayer {
	return &BertLayer{
		Attention:    NewBertAttention(config, backend),
		Intermediate: NewBertIntermediate(config, backend),
		Output:       NewBertOutput(config, backend),
	}
}

func (l *BertLayer) Forward(hiddenStates device.Tensor) device.Tensor {
	selfAttention := l.Attention.Forward(hiddenStates)
	intermediate := l.Intermediate.Forward(selfAttention)
	return l.Output.Forward(intermediate, selfAttention)
}

func (l *BertLayer) ForwardBatch(hiddenStates device.Tensor, lengths []int) device.Tensor {
	selfAttention := l.Attention.ForwardBatch(hiddenStates, lengths)
	intermediate := l.Intermediate.ForwardBatch(selfAttention)
	return l.Output.ForwardBatch(intermediate, selfAttention)
}

// BertPooler extracts the [CLS] representation.
type BertPooler struct {
	Backend device.Backend
	Dense   device.Tensor
	Bias    []float64
}

func NewBertPooler(config BertConfig, backend device.Backend) *BertPooler {
	return &BertPooler{
		Backend: backend,
		Dense:   backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Bias:    make([]float64, config.HiddenSize),
	}
}

func (p *BertPooler) Forward(hiddenStates device.Tensor) device.Tensor {
	// Take [CLS] token (first token at index 0)
	// clsToken is 1xH matrix
	// Take [CLS] token (first token at index 0)
	// clsToken is 1xH matrix
	_, h := hiddenStates.Dims()
	clsToken := hiddenStates.Slice(0, 1, 0, h)
	
	r, _ := p.Dense.Dims()
	output := p.Backend.NewTensor(1, r, nil)
	
	// output = clsToken * Dense + Bias
	// clsToken is 1xH. Dense is HxH. Output is 1xH.
	// Wait, standard is Dense * Input^T?
	// NewBertPooler creates Dense HxH.
	// Original code: output.MulVec(p.Dense, clsVec).
	// If p.Dense is HxH, and clsVec is H. Result is H.
	// If we use matrices: 1xH * HxH -> 1xH.
	// So output.Mul(clsToken, p.Dense).
	
	output.Mul(clsToken, p.Dense)
	
	output.Mul(clsToken, p.Dense)
	
	// Add bias and Tanh
	if len(p.Bias) > 0 {
		output.AddBias(p.Bias)
	}
	output.Tanh()
	
	return output
}

func (p *BertPooler) ForwardBatch(output device.Tensor, lengths []int) device.Tensor {
	batchSize := len(lengths)
	_, c := output.Dims()
	
	// Construct indices for CLS tokens (index 0 of each sequence).
	// Sequences are stacked in `output`.
	// cls index for sequence i is sum(lengths[:i]).
	indices := make([]int, batchSize)
	current := 0
	for i, l := range lengths {
		indices[i] = current
		current += l
	}
	
	// Use Gather to extract CLS tokens efficiently on backend
	// Gather is part of Tensor interface
	clsStack := output.Gather(indices)
	
	// Compute Result = CLS * Dense
	// Result dims: (BatchSize, Hidden)
	// Dense dims: (Hidden, Hidden)
	// clsStack dims: (BatchSize, Hidden)
	
	result := p.Backend.NewTensor(batchSize, c, nil)
	result.Mul(clsStack, p.Dense)
	
	// Add bias and Tanh
	if len(p.Bias) > 0 {
		result.AddBias(p.Bias)
	}
	result.Tanh()
	
	// Check if clsStack needs pooling?
	// It was created by Gather (NewTensor). 
	// We should probably rely on backend to manage lifecycle or return it?
	// But here it's a temporary intermediate.
    // If backend uses pooling, we might want to release it.
    // But current Tensor interface doesn't support manual release (Go GC handles it).
    // So just let it go.
	
	return result
}

func sumInts(v []int) int {
	s := 0
	for _, x := range v {
		s += x
	}
	return s
}

// BertAttention handles multi-head self-attention.
type BertAttention struct {
	Self       *BertSelfAttention
	Output     *BertSelfOutput
}

func NewBertAttention(config BertConfig, backend device.Backend) *BertAttention {
	return &BertAttention{
		Self:   NewBertSelfAttention(config, backend),
		Output: NewBertSelfOutput(config, backend),
	}
}

func (a *BertAttention) Forward(hiddenStates device.Tensor) device.Tensor {
	selfOutput := a.Self.Forward(hiddenStates)
	return a.Output.Forward(selfOutput, hiddenStates)
}

func (a *BertAttention) ForwardBatch(hiddenStates device.Tensor, lengths []int) device.Tensor {
	selfOutput := a.Self.ForwardBatch(hiddenStates, lengths)
	return a.Output.ForwardBatch(selfOutput, hiddenStates)
}

type BertSelfAttention struct {
	Backend           device.Backend
	NumAttentionHeads int
	AttentionHeadSize int
	AllHeadSize       int

	Query device.Tensor
	Key   device.Tensor
	Value device.Tensor

	QueryBias []float64
	KeyBias   []float64
	ValueBias []float64
}

func NewBertSelfAttention(config BertConfig, backend device.Backend) *BertSelfAttention {
	return &BertSelfAttention{
		Backend:           backend,
		NumAttentionHeads: config.NumAttentionHeads,
		AttentionHeadSize: config.HiddenSize / config.NumAttentionHeads,
		AllHeadSize:       config.HiddenSize,
		Query:             backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Key:               backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Value:             backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		QueryBias:         make([]float64, config.HiddenSize),
		KeyBias:           make([]float64, config.HiddenSize),
		ValueBias:         make([]float64, config.HiddenSize),
	}
}

func (s *BertSelfAttention) Forward(hiddenStates device.Tensor) device.Tensor {
	r, _ := hiddenStates.Dims()
	
	// Q, K, V Projections - use backend buffers
	queryLayer := projectPooled(s.Backend, hiddenStates, s.Query, s.QueryBias)
	keyLayer := projectPooled(s.Backend, hiddenStates, s.Key, s.KeyBias)
	valueLayer := projectPooled(s.Backend, hiddenStates, s.Value, s.ValueBias)

	// attentionScores = Q * K^T / sqrt(d_k)
	attentionScores := s.Backend.GetTensor(r, r)
	keyLayerT := keyLayer.T()
	attentionScores.Mul(queryLayer, keyLayerT)
	
	scale := 1.0 / math.Sqrt(float64(s.AttentionHeadSize))
	attentionScores.Scale(scale)
	
	// Softmax in-place
	attentionScores.Softmax()
	
	// contextLayer = attentionScores * V
	contextLayer := s.Backend.GetTensor(r, s.AllHeadSize)
	contextLayer.Mul(attentionScores, valueLayer)
	
	// Return Q, K, V, attentionScores to pool
	s.Backend.PutTensor(queryLayer)
	s.Backend.PutTensor(keyLayer)
	s.Backend.PutTensor(valueLayer)
	s.Backend.PutTensor(attentionScores)
	
	return contextLayer
}

func (s *BertSelfAttention) ForwardBatch(hiddenStates device.Tensor, lengths []int) device.Tensor {
	r, c := hiddenStates.Dims()
	
	// 1. Project Q, K, V for the entire batch at once
	queryLayer := projectPooled(s.Backend, hiddenStates, s.Query, s.QueryBias)
	keyLayer := projectPooled(s.Backend, hiddenStates, s.Key, s.KeyBias)
	valueLayer := projectPooled(s.Backend, hiddenStates, s.Value, s.ValueBias)
	
	output := s.Backend.NewTensor(r, c, nil)
	
	// Fast path: uniform sequence lengths (common case in batched inference)
	// This allows computing attention for all sequences without per-seq loop
	allSameLength := true
	seqLen := lengths[0]
	for _, l := range lengths {
		if l != seqLen {
			allSameLength = false
			break
		}
	}
	
	batchSize := len(lengths)
	
	if allSameLength && s.Backend.Name() != "CPU" {
		// Optimized GPU path for uniform sequence lengths
		// Process each sequence efficiently with minimal allocation
		scale := 1.0 / math.Sqrt(float64(s.AttentionHeadSize))
		
		for i := 0; i < batchSize; i++ {
			start := i * seqLen
			endIdx := start + seqLen
			
			seqQ := queryLayer.Slice(start, endIdx, 0, c)
			seqK := keyLayer.Slice(start, endIdx, 0, c)
			seqV := valueLayer.Slice(start, endIdx, 0, c)
			
			// Compute Attention Scores (seqLen x seqLen)
			attentionScores := s.Backend.GetTensor(seqLen, seqLen)
			seqKT := seqK.T()
			attentionScores.Mul(seqQ, seqKT)
			
			attentionScores.Scale(scale)
			attentionScores.Softmax()
			
			// Context: (seqLen x hidden)
			outSlice := output.Slice(start, endIdx, 0, c)
			outSlice.Mul(attentionScores, seqV)
			
			s.Backend.PutTensor(attentionScores)
		}
	} else {
		// Variable length path or CPU path
		useParallel := s.Backend.Name() == "CPU"
		
		currentIdx := 0
		type job struct {
			start int
			len   int
		}
		jobs := make([]job, len(lengths))
		for i, l := range lengths {
			jobs[i] = job{start: currentIdx, len: l}
			currentIdx += l
		}
		
		computeAttention := func(start, length int) {
			endIdx := start + length
			
			seqQ := queryLayer.Slice(start, endIdx, 0, c)
			seqK := keyLayer.Slice(start, endIdx, 0, c)
			seqV := valueLayer.Slice(start, endIdx, 0, c)
			
			attentionScores := s.Backend.GetTensor(length, length)
			seqKT := seqK.T()
			attentionScores.Mul(seqQ, seqKT)
			
			scale := 1.0 / math.Sqrt(float64(s.AttentionHeadSize))
			attentionScores.Scale(scale)
			attentionScores.Softmax()
			
			seqContext := s.Backend.GetTensor(length, s.AllHeadSize)
			seqContext.Mul(attentionScores, seqV)
			
			outSlice := output.Slice(start, endIdx, 0, c)
			outSlice.Copy(seqContext)
			
			s.Backend.PutTensor(attentionScores)
			s.Backend.PutTensor(seqContext)
		}
		
		if useParallel {
			var wg sync.WaitGroup
			for _, j := range jobs {
				wg.Add(1)
				start, length := j.start, j.len
				go func() {
					defer wg.Done()
					computeAttention(start, length)
				}()
			}
			wg.Wait()
		} else {
			for _, j := range jobs {
				computeAttention(j.start, j.len)
			}
		}
	}
	
	// Return projected layers
	s.Backend.PutTensor(queryLayer)
	s.Backend.PutTensor(keyLayer)
	s.Backend.PutTensor(valueLayer)
	
	return output
}

type BertSelfOutput struct {
	Backend   device.Backend
	Dense     device.Tensor
	Bias      []float64
	LayerNorm *LayerNorm
}

func NewBertSelfOutput(config BertConfig, backend device.Backend) *BertSelfOutput {
	return &BertSelfOutput{
		Backend:   backend,
		Dense:     backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Bias:      make([]float64, config.HiddenSize),
		LayerNorm: NewLayerNorm(config.HiddenSize, backend),
	}
}

func (o *BertSelfOutput) Forward(hiddenStates, inputTensor device.Tensor) device.Tensor {
	hiddenStates = projectPooled(o.Backend, hiddenStates, o.Dense, o.Bias)
	// Residual connection in-place
	// hiddenStates.Add(inputTensor). But inputTensor might not be same backing store?
	// AddInPlace assume shapes match.
	hiddenStates.Add(inputTensor)
	return o.LayerNorm.Forward(hiddenStates)
}

func (o *BertSelfOutput) ForwardBatch(hiddenStates, inputTensor device.Tensor) device.Tensor {
	return o.Forward(hiddenStates, inputTensor)
}

type BertIntermediate struct {
	Backend device.Backend
	Dense   device.Tensor
	Bias    []float64
}

func NewBertIntermediate(config BertConfig, backend device.Backend) *BertIntermediate {
	return &BertIntermediate{
		Backend: backend,
		Dense:   backend.NewTensor(config.HiddenSize, config.IntermediateSize, nil),
		Bias:    make([]float64, config.IntermediateSize),
	}
}

func (i *BertIntermediate) Forward(hiddenStates device.Tensor) device.Tensor {
	hiddenStates = projectPooledInter(i.Backend, hiddenStates, i.Dense, i.Bias)
	hiddenStates.Gelu()
	return hiddenStates
}

func (i *BertIntermediate) ForwardBatch(hiddenStates device.Tensor) device.Tensor {
	// Point-wise operations, can process whole batch at once
	return i.Forward(hiddenStates)
}

type BertOutput struct {
	Backend   device.Backend
	Dense     device.Tensor
	Bias      []float64
	LayerNorm *LayerNorm
}

func NewBertOutput(config BertConfig, backend device.Backend) *BertOutput {
	return &BertOutput{
		Backend:   backend,
		Dense:     backend.NewTensor(config.IntermediateSize, config.HiddenSize, nil),
		Bias:      make([]float64, config.HiddenSize),
		LayerNorm: NewLayerNorm(config.HiddenSize, backend),
	}
}

func (o *BertOutput) Forward(hiddenStates, inputTensor device.Tensor) device.Tensor {
	hiddenStates = projectPooled(o.Backend, hiddenStates, o.Dense, o.Bias)
	// Residual connection in-place
	hiddenStates.Add(inputTensor)
	return o.LayerNorm.Forward(hiddenStates)
}

func (o *BertOutput) ForwardBatch(hiddenStates, inputTensor device.Tensor) device.Tensor {
	return o.Forward(hiddenStates, inputTensor)
}

// Helpers

func projectPooled(backend device.Backend, input, weight device.Tensor, bias []float64) device.Tensor {
	r, _ := input.Dims()
	_, wc := weight.Dims()
	
	output := backend.GetTensor(r, wc)
	output.Mul(input, weight)
	
	if bias != nil {
		output.AddBias(bias)
	}
	
	return output
}

// projectPooledInter uses the same pool strategy as standard project
func projectPooledInter(backend device.Backend, input, weight device.Tensor, bias []float64) device.Tensor {
	// Re-use logic
	return projectPooled(backend, input, weight, bias)
}

// Optimized helpers are now part of device.Tensor interface (Softmax, Gelu, etc.)
// Removed legacy helper functions: softmax, gelu, softmaxInPlace, addInPlace, geluInPlace.
