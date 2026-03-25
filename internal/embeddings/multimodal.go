package embeddings

import (
	"fmt"
	"image"
	"image/color"
)

type InputType int

const (
	InputText InputType = iota
	InputImage
)

type MultiModalInput struct {
	Type  InputType
	Text  string
	Image image.Image
}

type EmbeddingInput struct {
	Inputs []MultiModalInput
}

func (e *EmbeddingInput) IsMultiModal() bool {
	for _, inp := range e.Inputs {
		if inp.Type == InputImage {
			return true
		}
	}
	return false
}

type ImagePreprocessor struct {
	TargetSize int
	Mean       [3]float32
	Std        [3]float32
}

func NewImagePreprocessor(targetSize int) *ImagePreprocessor {
	return &ImagePreprocessor{
		TargetSize: targetSize,
		Mean:       [3]float32{0.48145466, 0.4578275, 0.40821073},
		Std:        [3]float32{0.26862954, 0.26130258, 0.27577711},
	}
}

func (p *ImagePreprocessor) Preprocess(img image.Image) []float32 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	resized := image.NewNRGBA(image.Rect(0, 0, p.TargetSize, p.TargetSize))

	for y := 0; y < p.TargetSize; y++ {
		for x := 0; x < p.TargetSize; x++ {
			srcX := x * w / p.TargetSize
			srcY := y * h / p.TargetSize
			c := img.At(srcX, srcY)
			resized.Set(x, y, c)
		}
	}

	pixels := make([]float32, 3*p.TargetSize*p.TargetSize)
	idx := 0
	for y := 0; y < p.TargetSize; y++ {
		for x := 0; x < p.TargetSize; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			pixels[idx] = (float32(r>>8)/65535.0 - p.Mean[0]) / p.Std[0]
			pixels[idx+1] = (float32(g>>8)/65535.0 - p.Mean[1]) / p.Std[1]
			pixels[idx+2] = (float32(b>>8)/65535.0 - p.Mean[2]) / p.Std[2]
			idx += 3
		}
	}
	return pixels
}

type VisionEncoder interface {
	EncodeImage(pixels []float32) ([]float32, error)
	GetEmbeddingDim() int
}

type ClipVisionConfig struct {
	ImageSize  int
	PatchSize  int
	HiddenSize int
	NumLayers  int
	NumHeads   int
	EmbedDim   int
}

func DefaultClipVisionConfig() ClipVisionConfig {
	return ClipVisionConfig{
		ImageSize:  224,
		PatchSize:  32,
		HiddenSize: 768,
		NumLayers:  12,
		NumHeads:   12,
		EmbedDim:   512,
	}
}

type clipVisionEncoder struct {
	config  ClipVisionConfig
	backend interface{}
}

func NewClipVisionEncoder(config ClipVisionConfig, backend interface{}) (VisionEncoder, error) {
	return &clipVisionEncoder{
		config:  config,
		backend: backend,
	}, nil
}

func (e *clipVisionEncoder) EncodeImage(pixels []float32) ([]float32, error) {
	return nil, ErrNotImplemented
}

func (e *clipVisionEncoder) GetEmbeddingDim() int {
	return e.config.EmbedDim
}

var ErrNotImplemented = fmt.Errorf("not implemented")

func init() {
	_ = color.NRGBA{}
}
