package embeddings

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestImagePreprocessor(t *testing.T) {
	preprocessor := NewImagePreprocessor(224)

	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{128, 128, 128, 255})
		}
	}

	pixels := preprocessor.Preprocess(img)

	if len(pixels) != 3*224*224 {
		t.Errorf("Expected %d pixels, got %d", 3*224*224, len(pixels))
	}
}

func TestImagePreprocessor_DifferentSizes(t *testing.T) {
	preprocessor := NewImagePreprocessor(224)

	testSizes := []int{100, 200, 300, 500}
	for _, size := range testSizes {
		img := image.NewRGBA(image.Rect(0, 0, size, size))
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				img.Set(x, y, color.RGBA{255, 0, 0, 255})
			}
		}

		pixels := preprocessor.Preprocess(img)
		if len(pixels) != 3*224*224 {
			t.Errorf("Size %d: Expected %d pixels, got %d", size, 3*224*224, len(pixels))
		}
	}
}

func TestMultiModalInput_IsMultiModal(t *testing.T) {
	input := EmbeddingInput{
		Inputs: []MultiModalInput{
			{Type: InputText, Text: "hello"},
		},
	}

	if input.IsMultiModal() {
		t.Error("Expected false for text-only input")
	}

	input2 := EmbeddingInput{
		Inputs: []MultiModalInput{
			{Type: InputText, Text: "hello"},
			{Type: InputImage},
		},
	}

	if !input2.IsMultiModal() {
		t.Error("Expected true for mixed input")
	}
}

func TestVisionEncoderConfig(t *testing.T) {
	config := DefaultClipVisionConfig()

	if config.ImageSize != 224 {
		t.Errorf("Expected ImageSize 224, got %d", config.ImageSize)
	}
	if config.PatchSize != 32 {
		t.Errorf("Expected PatchSize 32, got %d", config.PatchSize)
	}
	if config.HiddenSize != 768 {
		t.Errorf("Expected HiddenSize 768, got %d", config.HiddenSize)
	}
	if config.NumLayers != 12 {
		t.Errorf("Expected NumLayers 12, got %d", config.NumLayers)
	}
	if config.NumHeads != 12 {
		t.Errorf("Expected NumHeads 12, got %d", config.NumHeads)
	}
	if config.EmbedDim != 512 {
		t.Errorf("Expected EmbedDim 512, got %d", config.EmbedDim)
	}
}

func TestVisionEncoder_Dimensions(t *testing.T) {
	config := ClipVisionConfig{
		ImageSize:  224,
		PatchSize:  16,
		HiddenSize: 384,
		NumLayers:  12,
		NumHeads:   6,
		EmbedDim:   384,
	}

	encoder, err := NewClipVisionEncoder(config, nil)
	if err != nil {
		t.Fatalf("Failed to create encoder: %v", err)
	}

	if encoder.GetEmbeddingDim() != 384 {
		t.Errorf("Expected embedding dim 384, got %d", encoder.GetEmbeddingDim())
	}
}

func TestImagePreprocessor_Normalization(t *testing.T) {
	preprocessor := NewImagePreprocessor(16)

	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			img.Set(x, y, color.RGBA{255, 255, 255, 255})
		}
	}

	pixels := preprocessor.Preprocess(img)

	for i, p := range pixels {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
			t.Errorf("NaN/Inf detected at index %d", i)
		}
	}
}
