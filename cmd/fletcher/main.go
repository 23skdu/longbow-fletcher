package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/23skdu/longbow-fletcher/internal/client"
	"github.com/23skdu/longbow-fletcher/internal/embeddings"
	"github.com/23skdu/longbow-fletcher/internal/embeddings/model"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/fatih/color"
	"github.com/joho/godotenv"
	"github.com/spf13/cobra"
)

var (
	cyan    = color.New(color.FgCyan, color.Bold).SprintFunc()
	green   = color.New(color.FgGreen, color.Bold).SprintFunc()
	yellow  = color.New(color.FgYellow).SprintFunc()
	red     = color.New(color.FgRed, color.Bold).SprintFunc()
	magenta = color.New(color.FgMagenta).SprintFunc()
	white   = color.New(color.FgWhite).SprintFunc()
)

func printBanner() {
	banner := `
  ███████╗██╗     ███████╗████████╗ ██████╗██╗  ██╗███████╗██████╗ 
  ██╔════╝██║     ██╔════╝╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔══██╗
  █████╗  ██║     █████╗     ██║   ██║     ███████║█████╗  ██████╔╝
  ██╔══╝  ██║     ██╔══╝     ██║   ██║     ██╔══██║██╔══╝  ██╔══██╗
  ██║     ███████╗███████╗   ██║   ╚██████╗██║  ██║███████╗██║  ██║
  ╚═╝     ╚══════╝╚══════╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
`
	color.Cyan(banner)
	fmt.Println(magenta("  Vector Embeddings for Longbow"))
	fmt.Println()
}

func main() {
	_ = godotenv.Load()

	var (
		serverAddr  string
		datasetName string
		vocabPath   string
		weightsPath string
		text        string
		loremCount  int
		batchSize   int
	)

	rootCmd := &cobra.Command{
		Use:   "fletcher",
		Short: "Longbow Fletcher: Pure Go vector text embeddings CLI",
		Run: func(cmd *cobra.Command, args []string) {
			printBanner()
			totalStart := time.Now()

			if text == "" && loremCount <= 0 {
				fmt.Println(red("✗ Error:"), "Text must be provided via --text or --lorem")
				os.Exit(1)
			}
			if vocabPath == "" {
				fmt.Println(red("✗ Error:"), "Vocab path must be provided via --vocab")
				os.Exit(1)
			}

			// 1. Initialize Embedder
			fmt.Printf("%s Initializing embedder...\n", yellow("⟳"))
			initStart := time.Now()
			config := model.DefaultBertTinyConfig()
			embedder, err := embeddings.NewEmbedder(vocabPath, weightsPath, config)
			if err != nil {
				fmt.Printf("%s Failed to initialize embedder: %v\n", red("✗"), err)
				os.Exit(1)
			}
			fmt.Printf("%s Embedder ready (dim=%d) %s\n", green("✓"), config.HiddenSize, white(fmt.Sprintf("[%v]", time.Since(initStart).Round(time.Millisecond))))

			// 2. Prepare Texts
			var texts []string
			if loremCount > 0 {
				fmt.Printf("%s Generating %d Lorem Ipsum paragraphs...\n", yellow("⟳"), loremCount)
				genStart := time.Now()
				texts = embeddings.GenerateLorem(loremCount)
				fmt.Printf("%s Generated %d paragraphs %s\n", green("✓"), len(texts), white(fmt.Sprintf("[%v]", time.Since(genStart).Round(time.Millisecond))))
			} else {
				texts = []string{text}
			}

			// 3. Generate Embeddings with parallelism
			fmt.Printf("\n%s Generating embeddings (parallel)...\n", cyan("▶"))
			embedStart := time.Now()
			
			// Parallel embedding generation
			numWorkers := runtime.GOMAXPROCS(0)
			embeddings32 := make([][]float32, len(texts))
			
			// Progress tracking
			var completed int64
			progressInterval := int64(len(texts) / 10)
			if progressInterval < 1 {
				progressInterval = 1
			}
			
			// Worker function
			var wg sync.WaitGroup
			
			// Calculate batches
			type batchJob struct {
				startIdx int
				endIdx   int
			}
			
			// Channel for batch jobs
			numBatches := (len(texts) + batchSize - 1) / batchSize
			batchChan := make(chan batchJob, numBatches)
			
			for w := 0; w < numWorkers; w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for job := range batchChan {
						// Extract batch texts
						batchTexts := texts[job.startIdx:job.endIdx]
						
						// Generate embeddings for batch
						batchResults := embedder.EmbedBatch(batchTexts)
						
						// Process results
						for k, vec := range batchResults {
							vec32 := make([]float32, len(vec))
							for j, v := range vec {
								vec32[j] = float32(v)
							}
							embeddings32[job.startIdx+k] = vec32
						}
						
						done := atomic.AddInt64(&completed, int64(len(batchTexts)))
						if len(texts) > 100 && done%progressInterval < int64(batchSize) {
							elapsed := time.Since(embedStart)
							rate := float64(done) / elapsed.Seconds()
							fmt.Printf("  %s [%d/%d] %.1f vec/s (%d workers)\n", yellow("⟳"), done, len(texts), rate, numWorkers)
						}
					}
				}()
			}
			
			// Send work
			for i := 0; i < len(texts); i += batchSize {
				end := i + batchSize
				if end > len(texts) {
					end = len(texts)
				}
				batchChan <- batchJob{startIdx: i, endIdx: end}
			}
			close(batchChan)
			wg.Wait()
			
			embedElapsed := time.Since(embedStart)
			embedRate := float64(len(embeddings32)) / embedElapsed.Seconds()
			fmt.Printf("%s Generated %d embeddings %s\n", green("✓"), len(embeddings32), white(fmt.Sprintf("[%v, %.1f vec/s, %d workers]", embedElapsed.Round(time.Millisecond), embedRate, numWorkers)))

			// 4. Format as Arrow RecordBatch
			fmt.Printf("\n%s Building Arrow RecordBatch...\n", cyan("▶"))
			arrowStart := time.Now()
			builder := client.NewRecordBatchBuilder(memory.DefaultAllocator)
			record, err := builder.BuildRecordBatch(embeddings32)
			if err != nil {
				fmt.Printf("%s Failed to build Arrow record: %v\n", red("✗"), err)
				os.Exit(1)
			}
			defer record.Release()
			fmt.Printf("%s Arrow RecordBatch ready (%d rows) %s\n", green("✓"), record.NumRows(), white(fmt.Sprintf("[%v]", time.Since(arrowStart).Round(time.Millisecond))))

			// 5. Send to Longbow via Flight
			if serverAddr != "" {
				fmt.Printf("\n%s Connecting to Longbow at %s...\n", cyan("▶"), serverAddr)
				flightClient, err := client.NewFlightClient(serverAddr)
				if err != nil {
					fmt.Printf("%s Failed to connect: %v\n", red("✗"), err)
					os.Exit(1)
				}
				defer flightClient.Close()
				fmt.Printf("%s Connected\n", green("✓"))

				fmt.Printf("%s Sending data to dataset '%s'...\n", yellow("⟳"), datasetName)
				sendStart := time.Now()
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute) // Extended for large batches
				defer cancel()

				if err := flightClient.DoPut(ctx, datasetName, record); err != nil {
					fmt.Printf("%s Failed to send: %v\n", red("✗"), err)
					os.Exit(1)
				}
				sendElapsed := time.Since(sendStart)
				sendRate := float64(len(texts)) / sendElapsed.Seconds()
				fmt.Printf("%s Successfully sent %d embeddings to Longbow! %s\n", green("✓"), len(texts), white(fmt.Sprintf("[%v, %.1f vec/s]", sendElapsed.Round(time.Millisecond), sendRate)))
			} else {
				fmt.Printf("\n%s Output (first 5 dims, max 10 shown):\n", cyan("▶"))
				maxShow := 10
				if len(embeddings32) < maxShow {
					maxShow = len(embeddings32)
				}
				for i := 0; i < maxShow; i++ {
					dims := embeddings32[i]
					if len(dims) > 5 {
						dims = dims[:5]
					}
					fmt.Printf("  [%d] %v\n", i, dims)
				}
				if len(embeddings32) > maxShow {
					fmt.Printf("  ... and %d more\n", len(embeddings32)-maxShow)
				}
			}

			// Final summary
			totalElapsed := time.Since(totalStart)
			fmt.Println()
			fmt.Printf("%s\n", cyan("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
			fmt.Printf("  %s Total: %d vectors in %v\n", green("✓ Done!"), len(embeddings32), totalElapsed.Round(time.Millisecond))
			fmt.Printf("  %s Throughput: %.1f vectors/second\n", cyan("📊"), float64(len(embeddings32))/totalElapsed.Seconds())
			fmt.Printf("%s\n", cyan("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
		},
	}

	rootCmd.Flags().StringVar(&serverAddr, "server", "", "Longbow server address (e.g., localhost:3000)")
	rootCmd.Flags().StringVar(&datasetName, "dataset", "fletcher_dataset", "Target dataset name")
	rootCmd.Flags().StringVar(&vocabPath, "vocab", "vocab.txt", "Path to vocab.txt")
	rootCmd.Flags().StringVar(&weightsPath, "weights", "", "Path to model weights binary")
	rootCmd.Flags().StringVar(&text, "text", "", "Text to embed")
	rootCmd.Flags().IntVar(&loremCount, "lorem", 0, "Number of Lorem Ipsum paragraphs to generate and embed")
	rootCmd.Flags().IntVar(&batchSize, "batch-size", 64, "Batch size for embedding generation")

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
