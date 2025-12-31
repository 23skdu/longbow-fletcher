package model

import (
	"math"
	"math/rand"
	"sync"

	"github.com/23skdu/longbow-fletcher/internal/device"
)

type PositionEmbeddingType int

const (
	PositionalAbsolute PositionEmbeddingType = iota
	PositionalRoPE
)

// BertConfig holds the configuration for the BERT model.
type BertConfig struct {
	VocabSize             int
	HiddenSize            int
	NumHiddenLayers       int
	NumAttentionHeads     int
	IntermediateSize      int
	MaxPositionEmbeddings int
	Activation            device.ActivationType
	PositionEmbedding      PositionEmbeddingType
}

// DefaultBertTinyConfig returns the configuration for BERT-Tiny.
func DefaultBertTinyConfig() BertConfig {
	return BertConfig{
		VocabSize:             30522,
		HiddenSize:            128,
		NumHiddenLayers:       2,
		NumAttentionHeads:     2,
		IntermediateSize:      512,
		MaxPositionEmbeddings: 512,
		Activation:            device.ActivationGELU,
		PositionEmbedding:      PositionalAbsolute,
	}
}

// DefaultNomicConfig returns the configuration for nomic-embed-text-v1.5.
func DefaultNomicConfig() BertConfig {
	return BertConfig{
		VocabSize:             30522,
		HiddenSize:            768,
		NumHiddenLayers:       12,
		NumAttentionHeads:     12,
		IntermediateSize:      3072,
		MaxPositionEmbeddings: 8192,
		Activation:            device.ActivationSwiGLU,
		PositionEmbedding:      PositionalRoPE,
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
// Uses bulk CopyFromFloat32 for efficient GPU upload (especially FP16).
func xavierInit(m device.Tensor) {
	r, c := m.Dims()
	size := r * c
	limit := math.Sqrt(6.0 / float64(r+c))
	
	// Generate all random values in a single slice
	data := make([]float32, size)
	for i := range data {
		data[i] = float32((rand.Float64()*2 - 1) * limit)
	}
	
	// Bulk upload to GPU (single FP16 conversion pass)
	m.CopyFromFloat32(data)
}

// ForwardBatch performs the forward pass for a batch of sequences.
// inputIDs is flattened, lengths contains the length of each sequence.
func (m *BertModel) ForwardBatch(inputIDs []int, lengths []int) device.Tensor {
	embeddings := m.Embeddings.ForwardBatch(inputIDs, lengths)
	
	hiddenStates := m.Encoder.ForwardBatch(embeddings, lengths)
	// embeddings is released by Encoder.
	
	res := m.Pooler.ForwardBatch(hiddenStates, lengths)
	m.Backend.PutTensor(hiddenStates)
	
	// Ensure all GPU operations are finished and memory is coherent before return
	m.Backend.Synchronize()
	
	return res
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
	
	// 2. Gather Position Embeddings (Absolute)
	if e.Config.PositionEmbedding == PositionalAbsolute {
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
		e.Backend.PutTensor(posEmbeds)
	}
	
	// 3. Token Type Embeddings (Assume 0)
	typeIndices := make([]int, totalTokens)
	typeEmbeds := e.TokenTypeEmbeddings.Gather(typeIndices)
	embeddings.Add(typeEmbeds)
	e.Backend.PutTensor(typeEmbeds)
	
	// 4. Norm + Dropout
	output := e.LayerNorm.Forward(embeddings)
	
	// Dropout is identity for now, returning same tensor.
	output = e.Dropout.Forward(output)
	
	return output
}

// LayerNorm implements Layer Normalization.
type LayerNorm struct {
	Gamma device.Tensor
	Beta  device.Tensor
	Eps   float32
}

func NewLayerNorm(size int, backend device.Backend) *LayerNorm {
	// Create Gamma with 1s
	ones := make([]float32, size)
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
	Backend device.Backend
	Layers  []*BertLayer
}

func NewBertEncoder(config BertConfig, backend device.Backend) *BertEncoder {
	layers := make([]*BertLayer, config.NumHiddenLayers)
	for i := range layers {
		layers[i] = NewBertLayer(config, backend)
	}
	return &BertEncoder{
		Backend: backend,
		Layers:  layers,
	}
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
		nextStates := layer.ForwardBatch(hiddenStates, lengths)
		e.Backend.PutTensor(hiddenStates)
		hiddenStates = nextStates
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
	// hiddenStates is NOT released here because it's released by the caller (Encoder)

	intermediate := l.Intermediate.ForwardBatch(selfAttention)
	
	res := l.Output.ForwardBatch(intermediate, selfAttention)
	
	// Release intermediates
	// selfAttention was used by Intermediate and Output, now done.
	l.Attention.Self.Backend.PutTensor(selfAttention)
	// intermediate is consumed by Output layer
	
	return res
}

// BertPooler extracts the [CLS] representation.
type BertPooler struct {
	Backend device.Backend
	Dense   device.Tensor
	Bias    device.Tensor
}

func NewBertPooler(config BertConfig, backend device.Backend) *BertPooler {
	return &BertPooler{
		Backend: backend,
		Dense:   backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Bias:    backend.NewTensor(1, config.HiddenSize, nil), // Zero initialized
	}
}

func (p *BertPooler) Forward(hiddenStates device.Tensor) device.Tensor {
	// Take [CLS] token (first token at index 0)
	// clsToken is 1xH matrix
	// Take [CLS] token (first token at index 0)
	// clsToken is 1xH matrix
	_, h := hiddenStates.Dims()
	clsToken := hiddenStates.Slice(0, 1, 0, h)
	
	// output = clsToken * Dense + Bias + Tanh
	output := p.Dense.LinearActivation(clsToken, p.Dense, p.Bias, device.ActivationTanh)
	
	return output
}

func (p *BertPooler) ForwardBatch(output device.Tensor, lengths []int) device.Tensor {
	batchSize := len(lengths)

	
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
	
	// Compute Result = CLS * Dense + Bias + Tanh
	result := p.Dense.LinearActivation(clsStack, p.Dense, p.Bias, device.ActivationTanh)
	p.Backend.PutTensor(clsStack)
	
	// Check if clsStack needs pooling?
	// It was created by Gather (NewTensor). 
	// We should probably rely on backend to manage lifecycle or return it?
	// But here it's a temporary intermediate.
    // If backend uses pooling, we might want to release it.
    // But current Tensor interface doesn't support manual release (Go GC handles it).
    // So just let it go.
	
	return result
}



// BertAttention handles multi-head self-attention.
type BertAttention struct {
	Config     BertConfig
	Self       *BertSelfAttention
	Output     *BertSelfOutput
}

func NewBertAttention(config BertConfig, backend device.Backend) *BertAttention {
	return &BertAttention{
		Config: config,
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
	res := a.Output.ForwardBatch(selfOutput, hiddenStates)
	// selfOutput is released by Output.ForwardBatch
	return res
}

type BertSelfAttention struct {
	Backend           device.Backend
	Config            BertConfig
	NumAttentionHeads int
	AttentionHeadSize int
	AllHeadSize       int

	Query device.Tensor
	Key   device.Tensor
	Value device.Tensor

	QueryBias device.Tensor
	KeyBias   device.Tensor
	ValueBias device.Tensor
}

func NewBertSelfAttention(config BertConfig, backend device.Backend) *BertSelfAttention {
	return &BertSelfAttention{
		Backend:           backend,
		Config:            config,
		NumAttentionHeads: config.NumAttentionHeads,
		AttentionHeadSize: config.HiddenSize / config.NumAttentionHeads,
		AllHeadSize:       config.HiddenSize,
		Query:             backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Key:               backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Value:             backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		QueryBias:         backend.NewTensor(1, config.HiddenSize, nil),
		KeyBias:           backend.NewTensor(1, config.HiddenSize, nil),
		ValueBias:         backend.NewTensor(1, config.HiddenSize, nil),
	}
}

func (s *BertSelfAttention) Forward(hiddenStates device.Tensor) device.Tensor {
	r, _ := hiddenStates.Dims()
	
	// Q, K, V Projections - use backend buffers
	queryLayer := s.Query.Linear(hiddenStates, s.Query, s.QueryBias)
	keyLayer := s.Key.Linear(hiddenStates, s.Key, s.KeyBias)
	valueLayer := s.Value.Linear(hiddenStates, s.Value, s.ValueBias)

	// Context Layer to accumulate heads
	contextLayer := s.Backend.NewTensor(r, s.AllHeadSize, nil) // Zero initialized

	scale := 1.0 / float32(math.Sqrt(float64(s.AttentionHeadSize)))

	// Multi-Head Attention Loop
	for h := 0; h < s.NumAttentionHeads; h++ {
		start := h * s.AttentionHeadSize
		end := start + s.AttentionHeadSize
		
		// Slice returns a COPY in current backend implementation
		qHead := queryLayer.Slice(0, r, start, end)
		kHead := keyLayer.Slice(0, r, start, end)
		vHead := valueLayer.Slice(0, r, start, end)
		
		// scores = Q * K^T
		scores := s.Backend.GetTensor(r, r)
		kHeadT := kHead.T()
		scores.Mul(qHead, kHeadT)
		scores.Scale(scale)
		scores.Softmax()
		
		// ctx = scores * V
		ctxHead := s.Backend.GetTensor(r, s.AttentionHeadSize)
		ctxHead.Mul(scores, vHead)
		
		// Copy ctxHead into contextLayer manually
		// TODO: Add Paste/SetSlice to Tensor interface for performance
		for i := 0; i < r; i++ {
			for j := 0; j < s.AttentionHeadSize; j++ {
				val := ctxHead.At(i, j)
				contextLayer.Set(i, start+j, val)
			}
		}
		
		// Cleanup intermediate tensors
		s.Backend.PutTensor(qHead) // slice created new tensor
		s.Backend.PutTensor(kHead)
		s.Backend.PutTensor(vHead)
		s.Backend.PutTensor(scores) // pooled
		s.Backend.PutTensor(ctxHead) 
	}
	
	// Return Q, K, V to pool
	s.Backend.PutTensor(queryLayer)
	s.Backend.PutTensor(keyLayer)
	s.Backend.PutTensor(valueLayer)
	
	return contextLayer
}

func (s *BertSelfAttention) ForwardBatch(hiddenStates device.Tensor, lengths []int) device.Tensor {
	_, c := hiddenStates.Dims()
	
	// 1. Project Q, K, V for the entire batch at once
	queryLayer := s.Query.Linear(hiddenStates, s.Query, s.QueryBias)
	keyLayer := s.Key.Linear(hiddenStates, s.Key, s.KeyBias)
	valueLayer := s.Value.Linear(hiddenStates, s.Value, s.ValueBias)
	
	// Apply RoPE if configured
	if s.Config.PositionEmbedding == PositionalRoPE {
		batchSize := len(lengths)
		seqLen := lengths[0] // Assume uniform for RoPE optimization or handle variable
		s.Backend.Synchronize() // Ensure linear projections are done
		queryLayer.ApplyRoPE(batchSize, seqLen, s.NumAttentionHeads, s.AttentionHeadSize)
		keyLayer.ApplyRoPE(batchSize, seqLen, s.NumAttentionHeads, s.AttentionHeadSize)
	}

	var output device.Tensor
	
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
	
	if allSameLength {
		// Optimized GPU/Graph path for uniform sequence lengths
		scale := 1.0 / float32(math.Sqrt(float64(s.AttentionHeadSize)))
		
		// Use Fused Attention Graph (Batch MatMul + Softmax + Context)
		// Returns (Batch*Seq, Hidden)
		contextLayer := queryLayer.Attention(queryLayer, keyLayer, valueLayer, batchSize, seqLen, s.NumAttentionHeads, scale)
		
		// Result is the output
		output = contextLayer
	} else {
		// Variable length path or CPU path
		useParallel := s.Backend.Name() == "CPU"
		
		// Initialize output tensor
		r, _ := hiddenStates.Dims()
		output = s.Backend.NewTensor(r, c, nil)
		
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
			
			// Compute self attention for this sequence manually with MHA
			for h := 0; h < s.NumAttentionHeads; h++ {
				headStart := h * s.AttentionHeadSize
				headEnd := headStart + s.AttentionHeadSize
				
				seqQ := queryLayer.Slice(start, endIdx, headStart, headEnd)
				seqK := keyLayer.Slice(start, endIdx, headStart, headEnd)
				seqV := valueLayer.Slice(start, endIdx, headStart, headEnd)
				
				attentionScores := s.Backend.GetTensor(length, length)
				seqKT := seqK.T()
				attentionScores.Mul(seqQ, seqKT)
				
				scale := 1.0 / float32(math.Sqrt(float64(s.AttentionHeadSize)))
				attentionScores.Scale(scale)
				attentionScores.Softmax()
				
				seqContext := s.Backend.GetTensor(length, s.AttentionHeadSize)
				seqContext.Mul(attentionScores, seqV)
				
				// Copy back to output (manual Set loop)
				// output[start:endIdx, headStart:headEnd] = seqContext
				for i := 0; i < length; i++ {
					for j := 0; j < s.AttentionHeadSize; j++ {
						output.Set(start+i, headStart+j, seqContext.At(i, j))
					}
				}
				
				s.Backend.PutTensor(seqQ)
				s.Backend.PutTensor(seqK)
				s.Backend.PutTensor(seqV)
				s.Backend.PutTensor(attentionScores)
				s.Backend.PutTensor(seqContext)
			}
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
	Bias      device.Tensor
	LayerNorm *LayerNorm
}

func NewBertSelfOutput(config BertConfig, backend device.Backend) *BertSelfOutput {
	return &BertSelfOutput{
		Backend:   backend,
		Dense:     backend.NewTensor(config.HiddenSize, config.HiddenSize, nil),
		Bias:      backend.NewTensor(1, config.HiddenSize, nil),
		LayerNorm: NewLayerNorm(config.HiddenSize, backend),
	}
}

func (o *BertSelfOutput) Forward(hiddenStates, inputTensor device.Tensor) device.Tensor {
	projected := o.Dense.Linear(hiddenStates, o.Dense, o.Bias)
	// input hiddenStates is no longer needed after projection
	o.Backend.PutTensor(hiddenStates)

	// Residual connection in-place
	projected.Add(inputTensor)
	return o.LayerNorm.Forward(projected)
}

func (o *BertSelfOutput) ForwardBatch(hiddenStates, inputTensor device.Tensor) device.Tensor {
	return o.Forward(hiddenStates, inputTensor)
}

type BertIntermediate struct {
	Backend device.Backend
	Config  BertConfig
	Dense   device.Tensor
	Bias    device.Tensor
}

func NewBertIntermediate(config BertConfig, backend device.Backend) *BertIntermediate {
	interSize := config.IntermediateSize
	if config.Activation == device.ActivationSwiGLU {
		interSize = config.IntermediateSize * 2 // Fused Gate + Up
	}
	return &BertIntermediate{
		Backend: backend,
		Config:  config,
		Dense:   backend.NewTensor(config.HiddenSize, interSize, nil),
		Bias:    backend.NewTensor(1, interSize, nil),
	}
}

func (i *BertIntermediate) Forward(hiddenStates device.Tensor) device.Tensor {
	res := i.Dense.LinearActivation(hiddenStates, i.Dense, i.Bias, i.Config.Activation)
	// input hiddenStates is no longer needed after projection
	// But wait! hiddenStates is owned by the caller (self-attention output).
	// We handle that in BertLayer.
	return res
}

func (i *BertIntermediate) ForwardBatch(hiddenStates device.Tensor) device.Tensor {
	// Point-wise operations, can process whole batch at once
	return i.Forward(hiddenStates)
}

type BertOutput struct {
	Backend   device.Backend
	Dense     device.Tensor
	Bias      device.Tensor
	LayerNorm *LayerNorm
}

func NewBertOutput(config BertConfig, backend device.Backend) *BertOutput {
	interSize := config.IntermediateSize
	// BertOutput always takes the REDUCED intermediate size as input, 
	// even if SwiGLU used 2x internally.
	return &BertOutput{
		Backend:   backend,
		Dense:     backend.NewTensor(interSize, config.HiddenSize, nil),
		Bias:      backend.NewTensor(1, config.HiddenSize, nil),
		LayerNorm: NewLayerNorm(config.HiddenSize, backend),
	}
}

func (o *BertOutput) Forward(hiddenStates, inputTensor device.Tensor) device.Tensor {
	projected := o.Dense.Linear(hiddenStates, o.Dense, o.Bias)
	// input hiddenStates (intermediate output) is no longer needed
	o.Backend.PutTensor(hiddenStates)

	// Residual connection in-place
	projected.Add(inputTensor)
	return o.LayerNorm.Forward(projected)
}

func (o *BertOutput) ForwardBatch(hiddenStates, inputTensor device.Tensor) device.Tensor {
	return o.Forward(hiddenStates, inputTensor)
}

// Helpers



// projectPooledInter uses the same pool strategy as standard project


// Optimized helpers are now part of device.Tensor interface (Softmax, Gelu, etc.)
// Removed legacy helper functions: softmax, gelu, softmaxInPlace, addInPlace, geluInPlace.
